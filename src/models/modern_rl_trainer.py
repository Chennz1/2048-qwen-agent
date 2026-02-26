"""
现代RL算法实现 - ReST (迭代式自训练)

为什么用ReST而不是REINFORCE:
- ✅ 2020年代算法，AlphaGo/ChatGPT时代的技术
- ✅ 自我对弈，不需要外部演示
- ✅ 迭代改进，每一轮都比上一轮强
- ✅ OpenAI、DeepMind在用

ReST = Reinforcement Learning via Self-Play + 自训练
类似AlphaGo的方法，但适配到LLM

优化支持:
- Unsloth + bitsandbytes + Flash Attention 2
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import json

# ⚠️ 重要：Unsloth必须在transformers之前导入
try:
    from unsloth import FastModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset

from src.envs.game_2048 import Game2048, parse_action_from_text, ACTION_MAP
from src.data.prompting import build_messages, format_inference_prompt, format_sample_text
from src.utils.monitoring import IterationMonitor, normalize_monitor_backend, report_to_list

try:
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    from trl import SFTTrainer, SFTConfig
    TRL_AVAILABLE = True
except ImportError:
    SFTTrainer = None
    SFTConfig = None
    TRL_AVAILABLE = False

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    PEFT_AVAILABLE = False


@dataclass
class ReSTConfig:
    """ReST训练配置"""
    # 迭代配置
    num_iterations: int = 5          # 迭代轮数
    games_per_iteration: int = 1000  # 每轮生成的游戏数

    # SFT配置（每轮用新数据微调）
    sft_epochs: int = 1              # 每轮SFT的epoch数
    learning_rate: float = 5e-5
    batch_size: int = 8

    # 数据筛选（只保留好的对局）
    top_k_ratio: float = 0.5         # 保留前50%的对局
    min_score_threshold: int = 500   # 最低分数阈值

    # 评估
    eval_games: int = 100            # 每轮评估游戏数


class GamePlayer:
    """游戏玩家 - 使用LLM策略玩游戏"""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
    ):
        self.model = model
        self.tokenizer = tokenizer

    def play_game(
        self,
        max_steps: int = 1000,
        seed: Optional[int] = None
    ) -> Dict:
        """
        玩一局游戏

        Returns:
            游戏信息字典，包含:
            - states: 状态列表
            - actions: 动作列表
            - scores: 分数列表
            - final_score: 最终分数
            - max_tile: 最大方块
            - num_steps: 步数
        """
        game = Game2048(seed=seed)
        game.reset()

        states = []
        actions = []
        scores = []

        for step in range(max_steps):
            state = game._get_state()

            # 模型选择动作
            action = self._choose_action(state)

            # 执行动作
            _, _, done, _ = game.step(action)

            # 记录
            states.append(state)
            actions.append(action)
            scores.append(game.score)

            if done:
                break

        return {
            'states': states,
            'actions': actions,
            'scores': scores,
            'final_score': game.score,
            'max_tile': game.get_max_tile(),
            'num_steps': len(states),
        }

    def _choose_action(self, state_text: str) -> int:
        """使用模型选择动作（使用 chat template + thinking 模式）

        使用与训练数据相同的 prompt 格式，确保一致性。
        官方最佳实践：
        - Thinking模式: Temperature=0.6, TopP=0.95, TopK=20
        - 必须使用采样 (do_sample=True)
        """
        prompt = format_inference_prompt(
            tokenizer=self.tokenizer,
            state_text=state_text,
            use_thinking=True,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,  # thinking模式需要更多token
                temperature=0.6,  # thinking模式最佳实践
                top_p=0.95,
                top_k=20,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        action = parse_action_from_text(response)
        return action


class ReSTTrainer:
    """
    ReST训练器 - 迭代式自训练

    算法流程:
    1. 用当前模型玩游戏
    2. 筛选高质量对局
    3. 用高质量对局微调模型
    4. 重复，模型越来越强
    """

    def __init__(
        self,
        base_model: str,
        output_dir: str = "./checkpoints/rest",
        config: Optional[ReSTConfig] = None,
        use_wandb: bool = False,
        monitor_backend: str = "none",
        # 优化选项 (默认开启)
        use_unsloth: bool = True,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        use_flash_attn: bool = False,
    ):
        """
        初始化ReST训练器

        Args:
            base_model: 基座模型路径（可以是SFT后的模型）
            output_dir: 输出目录
            config: ReST配置
            use_wandb: 是否使用WandB
            use_unsloth: 是否使用Unsloth优化
            load_in_4bit: 是否使用4-bit量化
            load_in_8bit: 是否使用8-bit量化
            use_flash_attn: 是否使用Flash Attention 2
        """
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.config = config or ReSTConfig()
        self.monitor_backend = normalize_monitor_backend(
            monitor_backend,
            use_wandb=use_wandb,
        )
        self.use_wandb = self.monitor_backend == "wandb"
        self.use_unsloth = use_unsloth
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.use_flash_attn = use_flash_attn
        self.monitor = IterationMonitor(
            backend=self.monitor_backend,
            log_dir=str(Path(output_dir) / "logs"),
        )

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检查flash-attn是否可用
        try:
            import flash_attn
            FLASH_ATTN_AVAILABLE = True
        except ImportError:
            FLASH_ATTN_AVAILABLE = False

        # 显示优化配置
        print("\n" + "=" * 70)
        print("⚡ ReST优化配置")
        print("=" * 70)
        print(f"📈 监控后端: {self.monitor_backend}")
        use_optimized = use_unsloth or load_in_4bit or load_in_8bit or use_flash_attn

        if use_unsloth:
            if UNSLOTH_AVAILABLE:
                if not FLASH_ATTN_AVAILABLE:
                    print("⚠️ Unsloth需要flash-attn，回退到标准模式")
                    print("   提示：安装flash-attn可获得更好性能（可选）")
                    self.use_unsloth = False
                else:
                    print("✅ Unsloth: 启用 (3-5x 加速)")
            else:
                print("⚠️ Unsloth: 未安装，回退到标准模式")
                self.use_unsloth = False

        if load_in_4bit:
            if BNB_AVAILABLE:
                print("✅ 4-bit量化: 启用")
            else:
                print("⚠️ bitsandbytes未安装，无法使用4-bit量化")
                self.load_in_4bit = False

        if load_in_8bit:
            if BNB_AVAILABLE:
                print("✅ 8-bit量化: 启用")
            else:
                print("⚠️ bitsandbytes未安装，无法使用8-bit量化")
                self.load_in_8bit = False

        if use_flash_attn and FLASH_ATTN_AVAILABLE:
            print("✅ Flash Attention 2: 启用")

        if not use_optimized:
            print("ℹ️ 使用标准训练模式")

        print("=" * 70)

        # 加载初始模型
        print(f"\n📦 加载基座模型: {base_model}")

        # 使用Unsloth（只有当flash-attn可用时）
        if self.use_unsloth and UNSLOTH_AVAILABLE:
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=base_model,
                load_in_4bit=self.load_in_4bit,
                max_seq_length=2048,
                device_map="auto",
            )
        else:
            # 标准加载
            bnb_config = None
            if self.load_in_4bit and BNB_AVAILABLE:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif self.load_in_8bit and BNB_AVAILABLE:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
            }

            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            if self.use_flash_attn:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            self.model = AutoModelForCausalLM.from_pretrained(
                base_model,
                **model_kwargs
            )

            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 训练历史
        self.history = {
            'iteration': [],
            'mean_score': [],
            'max_tile': [],
            'max_score': [],
        }

        print("✅ ReST训练器初始化完成\n")

    @staticmethod
    def _is_lora_adapter_checkpoint(model_path: str) -> bool:
        """判断路径是否为 LoRA adapter 检查点。"""
        return (Path(model_path) / "adapter_config.json").exists()

    @staticmethod
    def _resolve_base_model_from_adapter(adapter_path: str) -> str:
        """从 adapter_config.json 解析 LoRA 对应的基座模型。"""
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            raise ValueError(f"Not a LoRA adapter checkpoint: {adapter_path}")

        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)

        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(
                f"Cannot find 'base_model_name_or_path' in {adapter_config_path}"
            )
        return base_model

    def _load_model_for_generation(self, model_path: str):
        """加载用于对弈生成数据的模型（支持 LoRA adapter 路径）。"""
        if self._is_lora_adapter_checkpoint(model_path):
            if not PEFT_AVAILABLE:
                raise RuntimeError(
                    "`peft` is required to load LoRA checkpoints. Please install `peft>=0.12.0`."
                )
            base_model_path = self._resolve_base_model_from_adapter(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            return PeftModel.from_pretrained(
                base_model,
                model_path,
                is_trainable=False,
            )

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def _load_model_for_finetune(self, model_path: str):
        """加载用于下一轮 SFT 的模型。

        如果输入是 LoRA adapter，则先加载 base+adapter 并 merge，
        这样下一轮始终从可直接训练的完整模型权重继续。
        """
        if self._is_lora_adapter_checkpoint(model_path):
            if not PEFT_AVAILABLE:
                raise RuntimeError(
                    "`peft` is required to load LoRA checkpoints. Please install `peft>=0.12.0`."
                )
            base_model_path = self._resolve_base_model_from_adapter(model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            lora_model = PeftModel.from_pretrained(
                base_model,
                model_path,
                is_trainable=False,
            )
            return lora_model.merge_and_unload()

        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    def generate_iteration_data(
        self,
        model_path: str,
        num_games: int,
    ) -> Tuple[List[Dict], Dict]:
        """
        生成一轮迭代的数据

        Args:
            model_path: 当前模型路径
            num_games: 生成游戏数

        Returns:
            (games, stats)
        """
        print(f"🎮 生成 {num_games} 局游戏...")

        # 加载模型
        model = self._load_model_for_generation(model_path)

        player = GamePlayer(model, self.tokenizer)

        games = []
        scores = []
        max_tiles = []

        for i in tqdm(range(num_games), desc="生成游戏"):
            game_info = player.play_game(seed=i)
            games.append(game_info)
            scores.append(game_info['final_score'])
            max_tiles.append(game_info['max_tile'])

        stats = {
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': int(np.max(scores)),
            'mean_max_tile': float(np.mean(max_tiles)),
            'max_tile': int(np.max(max_tiles)),
        }

        print(f"  平均分数: {stats['mean_score']:.2f}")
        print(f"  最大方块: {stats['mean_max_tile']:.2f}")

        return games, stats

    def filter_games(
        self,
        games: List[Dict],
        top_k_ratio: float = 0.5,
        min_score: int = 500,
    ) -> List[Dict]:
        """
        筛选高质量对局

        Args:
            games: 游戏列表
            top_k_ratio: 保留前多少比例
            min_score: 最低分数阈值

        Returns:
            筛选后的游戏列表
        """
        # 按分数排序
        sorted_games = sorted(games, key=lambda x: x['final_score'], reverse=True)

        # 取top-k
        n_keep = max(int(len(games) * top_k_ratio), 1)
        top_games = sorted_games[:n_keep]

        # 过滤低分
        filtered = [g for g in top_games if g['final_score'] >= min_score]

        print(f"  筛选: {len(games)} → {len(filtered)} 局")
        print(f"  阈值: top {top_k_ratio*100}%, 最低分数 {min_score}")

        return filtered

    def games_to_training_data(
        self,
        games: List[Dict]
    ) -> Dataset:
        """
        将游戏转换为训练数据

        Args:
            games: 游戏列表

        Returns:
            训练数据集
        """
        data_samples = []

        for game in games:
            for state, action in zip(game['states'], game['actions']):
                action_name = ACTION_MAP[action] if isinstance(action, int) else action
                messages = build_messages(
                    state_text=state,
                    action=action_name,
                    use_thinking=False,
                )
                full_text = format_sample_text(
                    tokenizer=self.tokenizer,
                    messages=messages,
                    apply_chat_template=True,
                )
                data_samples.append({
                    "text": full_text,
                })

        return Dataset.from_list(data_samples)

    def finetune_on_data(
        self,
        train_data: Dataset,
        model_path: str,
        output_dir: str,
        num_epochs: int = 1,
    ) -> str:
        """
        在数据上微调模型

        Args:
            train_data: 训练数据
            model_path: 当前迭代的模型路径
            output_dir: 输出目录
            num_epochs: 训练轮数
        """
        if not TRL_AVAILABLE:
            raise RuntimeError("`trl` is required for ReST finetuning. Please install `trl>=0.12.0`.")
        print(f"🔧 微调模型...")

        model_for_train = self._load_model_for_finetune(model_path)

        # 配置
        sft_config = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            learning_rate=self.config.learning_rate,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="no",
            fp16=True,
            gradient_checkpointing=True,
            dataset_text_field="text",
            report_to=report_to_list(self.monitor_backend),
        )

        # LoRA配置
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # 训练
        trainer = SFTTrainer(
            model=model_for_train,
            args=sft_config,
            train_dataset=train_data,
            tokenizer=self.tokenizer,
            peft_config=lora_config,
        )

        trainer.train()

        # 保存
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # 额外保存一份可直接重载的完整模型，供下一轮 ReST 训练使用。
        merged_model_dir = Path(output_dir) / "merged_model"
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_model_dir))
        self.tokenizer.save_pretrained(str(merged_model_dir))

        with open(Path(output_dir) / "rest_checkpoint_info.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "adapter_path": str(output_dir),
                    "merged_model_path": str(merged_model_dir),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"  模型已保存到: {output_dir}")
        print(f"  可重载完整模型: {merged_model_dir}")

        return str(merged_model_dir)

    def evaluate(
        self,
        model_path: str,
        num_games: int = 100,
    ) -> Dict:
        """评估模型"""
        games, stats = self.generate_iteration_data(
            model_path, num_games
        )
        return stats

    def train(self):
        """主训练循环"""
        self.monitor.init(
            project="2048-game-rest",
            config=vars(self.config),
        )

        print("="*70)
        print("ReST 训练开始")
        print("="*70)
        print(f"迭代轮数: {self.config.num_iterations}")
        print(f"每轮游戏数: {self.config.games_per_iteration}")
        print(f"筛选比例: {self.config.top_k_ratio}")
        print()

        # 当前模型路径（初始是base_model）
        current_model = self.base_model

        for iteration in range(1, self.config.num_iterations + 1):
            print(f"\n{'='*70}")
            print(f"迭代 {iteration}/{self.config.num_iterations}")
            print(f"{'='*70}\n")

            # Step 1: 用当前模型生成数据
            print("Step 1: 生成游戏数据")
            games, gen_stats = self.generate_iteration_data(
                current_model,
                self.config.games_per_iteration,
            )

            # Step 2: 筛选高质量对局
            print("\nStep 2: 筛选高质量对局")
            filtered_games = self.filter_games(
                games,
                top_k_ratio=self.config.top_k_ratio,
                min_score=self.config.min_score_threshold,
            )

            if len(filtered_games) == 0:
                print("❌ 没有合格的游戏，跳过本轮训练")
                continue

            # Step 3: 转换为训练数据
            print("\nStep 3: 转换为训练数据")
            train_data = self.games_to_training_data(filtered_games)
            print(f"  训练样本数: {len(train_data)}")

            # Step 4: 微调模型
            print("\nStep 4: 微调模型")
            iter_output = self.output_dir / f"iteration_{iteration}"
            merged_model_path = self.finetune_on_data(
                train_data,
                current_model,
                str(iter_output),
                num_epochs=self.config.sft_epochs,
            )

            # Step 5: 评估新模型
            print("\nStep 5: 评估新模型")
            eval_stats = self.evaluate(
                str(iter_output),
                self.config.eval_games,
            )

            # 记录历史
            self.history['iteration'].append(iteration)
            self.history['mean_score'].append(eval_stats['mean_score'])
            self.history['max_tile'].append(eval_stats['mean_max_tile'])
            self.history['max_score'].append(eval_stats['max_score'])

            # WandB记录
            self.monitor.log(
                {
                    "iteration": iteration,
                    "mean_score": eval_stats['mean_score'],
                    "mean_max_tile": eval_stats['mean_max_tile'],
                    "max_score": eval_stats['max_score'],
                },
                step=iteration,
            )

            # 打印总结
            print(f"\n✅ 迭代 {iteration} 完成")
            print(f"  生成平均分: {gen_stats['mean_score']:.2f}")
            print(f"  评估平均分: {eval_stats['mean_score']:.2f}")
            print(f"  最大方块: {eval_stats['mean_max_tile']:.2f}")
            print(f"  最高分: {eval_stats['max_score']}")

            # 更新当前模型
            current_model = merged_model_path

        # 保存训练历史
        with open(self.output_dir / "training_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)

        self.monitor.finish()

        print("\n" + "="*70)
        print("🎉 ReST训练完成!")
        print("="*70)
        print(f"最终模型: {current_model}")
        print(f"训练历史: {self.output_dir / 'training_history.json'}")

        return current_model


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="ReST训练2048游戏模型")
    parser.add_argument("--base_model", type=str, default="./checkpoints/sft",
                        help="基座模型（SFT后的模型）")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/rest",
                        help="输出目录")
    parser.add_argument("--num_iterations", type=int, default=5,
                        help="迭代轮数")
    parser.add_argument("--games_per_iteration", type=int, default=1000,
                        help="每轮生成的游戏数")
    parser.add_argument("--top_k_ratio", type=float, default=0.5,
                        help="保留前多少比例的对局")
    parser.add_argument("--min_score", type=int, default=500,
                        help="最低分数阈值")
    parser.add_argument("--sft_epochs", type=int, default=1,
                        help="每轮SFT的epoch数")
    parser.add_argument("--no_wandb", action="store_true",
                        help="不使用WandB")
    parser.add_argument(
        "--monitor_backend",
        type=str,
        default=None,
        choices=["wandb", "tensorboard", "none"],
        help="监控后端 (wandb/tensorboard/none)",
    )

    # 优化选项
    parser.add_argument("--use_unsloth", action="store_true",
                        help="使用Unsloth优化 (3-5x加速)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="使用4-bit量化")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="使用8-bit量化")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="使用Flash Attention 2")

    args = parser.parse_args()

    # 配置
    config = ReSTConfig(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        top_k_ratio=args.top_k_ratio,
        min_score_threshold=args.min_score,
        sft_epochs=args.sft_epochs,
    )

    monitor_backend = normalize_monitor_backend(
        args.monitor_backend,
        no_wandb=args.no_wandb,
    )

    # 训练
    trainer = ReSTTrainer(
        base_model=args.base_model,
        output_dir=args.output_dir,
        config=config,
        use_wandb=not args.no_wandb,
        monitor_backend=monitor_backend,
        use_unsloth=args.use_unsloth,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        use_flash_attn=args.use_flash_attn,
    )

    trainer.train()


if __name__ == "__main__":
    main()
