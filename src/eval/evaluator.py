"""
Evaluation System for 2048 Game Model
Evaluates model performance on 2048 game.

优化支持:
- vLLM + INT8量化 (10-50x 推理加速)
"""

import argparse
import ast
import hashlib
import os
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

from src.envs.game_2048 import Game2048, ACTION_MAP, parse_action_from_text
from src.data_gen.prompting import format_inference_prompt

# Hugging Face mirror defaults (honor existing env if user already set it).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PeftModel = None
    PEFT_AVAILABLE = False

# 尝试导入 vLLM
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

try:
    from vllm.lora.request import LoRARequest
    VLLM_LORA_AVAILABLE = True
except ImportError:
    LoRARequest = None
    VLLM_LORA_AVAILABLE = False

THINKING_SAMPLING_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}

NON_THINKING_SAMPLING_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0.0,
}

RANDOM_EVAL_INIT_TILES_MIN = 1
RANDOM_EVAL_INIT_TILES_MAX = 4
RANDOM_EVAL_INIT_PROB_4 = 0.5


class Game2048Evaluator:
    """2048 game evaluator"""

    def __init__(
        self,
        model_path: str = None,
        base_model: str = "Qwen/Qwen3-1.7B",
        device: str = "auto",
        is_base_model: bool = False,
        use_thinking: bool = True,
        presence_penalty: float = 0.0,
        # 优化选项 (默认开启)
        use_vllm: bool = True,
        vllm_quantization: Optional[str] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to LoRA checkpoint (None for base model only)
            base_model: Base model name
            device: Device to use
            is_base_model: If True, only load base model without LoRA weights
            use_thinking: Whether to enable thinking mode prompt/template
            presence_penalty: Presence penalty for supported frameworks (0~2)
            use_vllm: Use vLLM for fast inference (10-50x speedup)
            vllm_quantization: vLLM quantization method (None means disable quantization)
            load_in_4bit: Use 4-bit quantization (standard loading)
            load_in_8bit: Use 8-bit quantization (standard loading)
        """
        self.use_thinking = bool(use_thinking)
        self.presence_penalty = float(np.clip(float(presence_penalty), 0.0, 2.0))
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.vllm_model = None
        self.vllm_lora_request = None
        self.model_type = "base_model" if is_base_model or model_path is None else "finetuned"

        # Evaluation policy: prefer BF16, avoid bitsandbytes quantization by default.
        if load_in_4bit or load_in_8bit:
            print("⚠️ 已忽略 load_in_4bit/load_in_8bit：评测默认使用 BF16，不使用 bitsandbytes 量化。")
            load_in_4bit = False
            load_in_8bit = False

        # 显示优化配置
        print("\n" + "=" * 70)
        print("⚡ 评估优化配置")
        print("=" * 70)

        if use_vllm:
            if VLLM_AVAILABLE:
                print(f"✅ vLLM: 启用 (10-50x 加速)")
                if vllm_quantization:
                    print(f"   量化: {vllm_quantization.upper()}")
            else:
                print("⚠️ vLLM未安装，回退到标准模式")
                self.use_vllm = False

        if not use_vllm:
            print("ℹ️ 使用标准评估模式")

        print("=" * 70)

        # 使用 vLLM
        if self.use_vllm:
            self._init_vllm(model_path, base_model, is_base_model, vllm_quantization)
        else:
            # 标准加载
            self._init_standard(model_path, base_model, device, is_base_model,
                              load_in_4bit, load_in_8bit)

    @staticmethod
    def _is_lora_adapter_checkpoint(model_path: str) -> bool:
        """判断路径是否为 LoRA adapter 检查点。"""
        if not model_path:
            return False
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

    @staticmethod
    def _resolve_lora_rank_from_adapter(adapter_path: str) -> Optional[int]:
        """从 adapter_config.json 解析 LoRA rank (r)。"""
        adapter_config_path = Path(adapter_path) / "adapter_config.json"
        if not adapter_config_path.exists():
            return None
        try:
            with open(adapter_config_path, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)
        except Exception:
            return None
        rank = adapter_config.get("r")
        if rank is None:
            return None
        try:
            rank_i = int(rank)
        except Exception:
            return None
        return rank_i if rank_i > 0 else None

    @staticmethod
    def _normalize_vllm_quantization(quantization: Optional[str]) -> Optional[str]:
        """Normalize user-facing quantization aliases to vLLM-supported names."""
        if not quantization:
            return None
        q = str(quantization).strip().lower()
        alias = {
            "int8": "bitsandbytes",
            "bnb": "bitsandbytes",
        }
        return alias.get(q, q)

    @staticmethod
    def _ensure_vllm_multiproc_spawn() -> None:
        """Avoid CUDA re-init crash in forked vLLM worker subprocesses."""
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    def _init_vllm(
        self,
        model_path: str,
        base_model: str,
        is_base_model: bool,
        quantization: Optional[str],
    ):
        """使用 vLLM 初始化模型"""
        self._ensure_vllm_multiproc_spawn()
        model_ref = base_model

        if is_base_model or not model_path:
            print(f"\n📦 使用 vLLM 加载基座模型: {base_model}")
        elif self._is_lora_adapter_checkpoint(model_path):
            model_ref = self._resolve_base_model_from_adapter(model_path)
            print(f"\n📦 使用 vLLM 加载基座模型: {model_ref}")
            print(f"   并挂载 LoRA adapter: {model_path}")
        else:
            model_ref = model_path
            print(f"\n📦 使用 vLLM 加载完整微调模型: {model_ref}")

        llm_kwargs = {
            "model": model_ref,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 1024,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
        }

        if model_path and (not is_base_model) and self._is_lora_adapter_checkpoint(model_path):
            if not VLLM_LORA_AVAILABLE:
                raise RuntimeError(
                    "Current vLLM build does not expose LoRARequest. "
                    "Please upgrade vLLM for LoRA support, or disable --use_vllm."
                )
            llm_kwargs["enable_lora"] = True
            lora_rank = self._resolve_lora_rank_from_adapter(model_path)
            if lora_rank is not None:
                llm_kwargs["max_lora_rank"] = max(16, lora_rank)
                print(f"   vLLM max_lora_rank={llm_kwargs['max_lora_rank']} (adapter r={lora_rank})")

        normalized_quant = self._normalize_vllm_quantization(quantization)
        if normalized_quant:
            if normalized_quant != (quantization or "").lower():
                print(f"   量化别名映射: {quantization} -> {normalized_quant}")
            llm_kwargs["quantization"] = normalized_quant

        try:
            self.vllm_model = LLM(**llm_kwargs)
        except Exception as exc:
            if normalized_quant and "quantization" in str(exc).lower():
                print(
                    f"⚠️ vLLM quantization='{normalized_quant}' 初始化失败，"
                    "回退到无量化 vLLM。"
                )
                llm_kwargs.pop("quantization", None)
                self.vllm_model = LLM(**llm_kwargs)
            elif "Cannot re-initialize CUDA in forked subprocess" in str(exc):
                raise RuntimeError(
                    "vLLM failed due to CUDA+fork multiprocessing. "
                    "Set VLLM_WORKER_MULTIPROC_METHOD=spawn (or use --no_vllm)."
                ) from exc
            else:
                raise
        self.tokenizer = self.vllm_model.get_tokenizer()

        if model_path and (not is_base_model) and self._is_lora_adapter_checkpoint(model_path):
            self.vllm_lora_request = LoRARequest("eval_adapter", 1, model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models should use left padding for batched generation.
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

        print("✅ vLLM 模型加载成功\n")

    def _init_standard(
        self,
        model_path: str,
        base_model: str,
        device: str,
        is_base_model: bool,
        load_in_4bit: bool,
        load_in_8bit: bool,
    ):
        """标准方式初始化模型"""
        is_adapter_checkpoint = bool(model_path) and self._is_lora_adapter_checkpoint(model_path)

        # 确定加载模式
        if is_base_model or model_path is None:
            print(f"\n🔵 加载基座模型: {base_model}")
            model_to_load = base_model
            tokenizer_to_load = base_model
            need_lora = False
        elif is_adapter_checkpoint:
            adapter_base_model = self._resolve_base_model_from_adapter(model_path)
            print(f"\n🟢 加载微调模型(LoRA): {model_path}")
            print(f"   基座模型: {adapter_base_model}")
            model_to_load = adapter_base_model
            tokenizer_to_load = adapter_base_model
            need_lora = True
        else:
            print(f"\n🟢 加载完整微调模型: {model_path}")
            model_to_load = model_path
            tokenizer_to_load = model_path
            need_lora = False

        # Load base model
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if use_bf16 else torch.float16,
            "device_map": device,
            "trust_remote_code": True,
        }
        print(f"   推理精度: {'bf16' if use_bf16 else 'fp16'}")

        self.model = AutoModelForCausalLM.from_pretrained(model_to_load, **model_kwargs)

        # Load LoRA weights if provided and not base model mode
        if need_lora and model_path and Path(model_path).exists():
            if not PEFT_AVAILABLE:
                raise RuntimeError(
                    "`peft` is required to load LoRA checkpoints. Install `peft>=0.12.0`."
                )
            self.model = PeftModel.from_pretrained(
                self.model,
                model_path,
                is_trainable=False
            )
            print(f"   LoRA 权重已加载: {model_path}")
        elif need_lora and model_path:
            print(f"   ⚠️  警告: LoRA 路径 {model_path} 不存在，仅使用基座模型")
            self.model_type = "base_model"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_to_load,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Decoder-only models should use left padding for batched generation.
        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

        self.model.eval()
        print("✅ 模型加载成功\n")

    def _resolve_sampling(self, temperature: Optional[float]) -> Dict[str, float]:
        defaults = THINKING_SAMPLING_DEFAULTS if self.use_thinking else NON_THINKING_SAMPLING_DEFAULTS
        resolved_temperature = defaults["temperature"] if temperature is None else max(float(temperature), 1e-3)
        return {
            "temperature": resolved_temperature,
            "top_p": defaults["top_p"],
            "top_k": defaults["top_k"],
            "min_p": defaults["min_p"],
        }

    def predict_action(
        self,
        state_text: str,
        temperature: Optional[float] = None
    ) -> int:
        """
        Predict action using the model.

        使用 chat template 格式，让模型自然停止。

        Args:
            state_text: Current game state as text
            temperature: Sampling temperature override (None means mode default)

        Returns:
            Action ID (0-3)
        """
        return self.predict_actions([state_text], temperature=temperature)[0]

    def predict_actions(
        self,
        state_texts: List[str],
        temperature: Optional[float] = None,
    ) -> List[int]:
        """Predict actions for a batch of states."""
        if not state_texts:
            return []
        if self.use_vllm:
            return self._predict_actions_vllm(state_texts, temperature)
        return self._predict_actions_standard(state_texts, temperature)

    def _predict_actions_vllm(
        self,
        state_texts: List[str],
        temperature: Optional[float],
    ) -> List[int]:
        """使用 vLLM 批量预测动作

        使用与训练数据相同的 prompt 格式，确保一致性。
        官方最佳实践：
        - Thinking模式: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
        - Non-thinking模式: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
        - 禁止贪心解码
        """
        sampling = self._resolve_sampling(temperature)
        prompts = [
            format_inference_prompt(
                tokenizer=self.tokenizer,
                state_text=state_text,
                use_thinking=self.use_thinking,
            )
            for state_text in state_texts
        ]

        # vLLM 批量推理 - 使用最佳实践参数
        sampling_params = SamplingParams(
            temperature=sampling["temperature"],
            top_p=sampling["top_p"],
            top_k=sampling["top_k"],
            min_p=sampling["min_p"],
            max_tokens=512 if self.use_thinking else 64,
            presence_penalty=self.presence_penalty,
        )

        if self.vllm_lora_request is not None:
            outputs = self.vllm_model.generate(
                prompts,
                sampling_params,
                lora_request=self.vllm_lora_request,
                use_tqdm=False,
            )
        else:
            outputs = self.vllm_model.generate(prompts, sampling_params, use_tqdm=False)

        actions: List[int] = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            actions.append(parse_action_from_text(text))
        return actions

    def _predict_actions_standard(
        self,
        state_texts: List[str],
        temperature: Optional[float],
    ) -> List[int]:
        """使用标准方式批量预测动作

        使用与训练数据相同的 prompt 格式，确保一致性。
        官方最佳实践：
        - Thinking模式: Temperature=0.6, TopP=0.95, TopK=20, MinP=0
        - Non-thinking模式: Temperature=0.7, TopP=0.8, TopK=20, MinP=0
        - 禁止贪心解码
        """
        sampling = self._resolve_sampling(temperature)
        prompts = [
            format_inference_prompt(
                tokenizer=self.tokenizer,
                state_text=state_text,
                use_thinking=self.use_thinking,
            )
            for state_text in state_texts
        ]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            generate_kwargs = {
                **inputs,
                "max_new_tokens": 768 if self.use_thinking else 64,
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "top_k": sampling["top_k"],
                "do_sample": True,  # 必须使用采样，禁止贪心解码
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            generation_config_dict = (
                self.model.generation_config.to_dict()
                if hasattr(self.model, "generation_config") and hasattr(self.model.generation_config, "to_dict")
                else {}
            )
            if "min_p" in generation_config_dict:
                generate_kwargs["min_p"] = sampling["min_p"]
            outputs = self.model.generate(**generate_kwargs)

        prompt_len = inputs["input_ids"].shape[1]
        actions: List[int] = []
        for seq in outputs:
            response = self.tokenizer.decode(
                seq[prompt_len:],
                skip_special_tokens=True,
            )
            actions.append(parse_action_from_text(response))

        return actions

    def evaluate(
        self,
        num_games: int = 100,
        max_steps: int = 1000,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        batch_size: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model performance.

        Args:
            num_games: Number of games to play
            max_steps: Maximum steps per game
            temperature: Sampling temperature override (None means mode default)
            seed: Global random seed for reproducibility
            batch_size: Number of parallel game states per forward pass
            verbose: Print progress

        Returns:
            Dictionary with evaluation metrics
        """
        scores = []
        max_tiles = []
        legal_move_ratios = []
        steps_list = []

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        batch_size = max(1, int(batch_size))
        completed = 0

        for start in range(0, num_games, batch_size):
            end = min(start + batch_size, num_games)
            records = []
            for game_idx in range(start, end):
                game_seed = self._derive_episode_seed(seed, game_idx)
                episode_rng = random.Random(game_seed)
                init_tiles = episode_rng.randint(
                    RANDOM_EVAL_INIT_TILES_MIN,
                    RANDOM_EVAL_INIT_TILES_MAX,
                )
                game = Game2048(seed=game_seed)
                game.reset(initial_tiles=init_tiles, prob_4=RANDOM_EVAL_INIT_PROB_4)
                records.append({
                    "game": game,
                    "game_idx": game_idx,
                    "done": False,
                    "total_moves": 0,
                    "legal_moves": 0,
                })

            for _ in range(max_steps):
                active_indices = [i for i, rec in enumerate(records) if not rec["done"]]
                if not active_indices:
                    break

                states = [records[i]["game"]._get_state() for i in active_indices]
                actions = self.predict_actions(states, temperature=temperature)

                for rec_idx, action in zip(active_indices, actions):
                    rec = records[rec_idx]
                    game = rec["game"]
                    was_done = rec["done"]
                    valid_actions = game.get_valid_actions()
                    is_legal = action in valid_actions
                    if is_legal:
                        rec["legal_moves"] += 1
                        _, _, done, _ = game.step(action)
                    else:
                        if valid_actions:
                            _, _, done, _ = game.step(valid_actions[0])
                        else:
                            done = True
                    rec["total_moves"] += 1
                    rec["done"] = bool(done)
                    if verbose and (not was_done) and rec["done"]:
                        print(f"Game {rec['game_idx']} finished with score={game.score}")

            for rec in records:
                game = rec["game"]
                total_moves = rec["total_moves"]
                scores.append(game.score)
                max_tiles.append(game.get_max_tile())
                legal_move_ratios.append(rec["legal_moves"] / total_moves if total_moves > 0 else 0)
                steps_list.append(total_moves)
                completed += 1
                if verbose and completed % 10 == 0:
                    print(f"Completed {completed}/{num_games} games")

        results = {
            'num_games': num_games,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': int(np.max(scores)),
            'min_score': int(np.min(scores)),
            'median_score': float(np.median(scores)),
            'mean_max_tile': float(np.mean(max_tiles)),
            'max_tile_reached': int(np.max(max_tiles)),
            'legal_move_rate': float(np.mean(legal_move_ratios)),
            'mean_steps': float(np.mean(steps_list)),
            'scores': [int(s) for s in scores],
            'max_tiles': [int(t) for t in max_tiles],
            'legal_move_ratios': [float(r) for r in legal_move_ratios],
            'steps': [int(s) for s in steps_list]
        }

        return results

    def evaluate_from_eval_set(
        self,
        eval_set_path: str,
        max_steps: int = 1000,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        batch_size: int = 1,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model from fixed eval-set snapshots (jsonl).

        Each line should contain at least:
        - state: board text
        Optional:
        - start_score
        - bucket
        """
        samples = self._load_eval_set(eval_set_path)
        if not samples:
            raise ValueError(f"Empty eval set: {eval_set_path}")

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        scores = []
        max_tiles = []
        legal_move_ratios = []
        steps_list = []
        bucket_counter: Dict[str, int] = {}

        batch_size = max(1, int(batch_size))
        completed = 0

        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start: start + batch_size]
            records = []
            for local_idx, sample in enumerate(batch_samples):
                idx = start + local_idx
                game_seed = self._derive_episode_seed(seed, idx, sample)
                game = self._init_game_from_sample(sample, seed=game_seed)
                bucket = sample.get("bucket", "unknown")
                bucket_counter[bucket] = bucket_counter.get(bucket, 0) + 1
                records.append({
                    "game": game,
                    "game_idx": idx,
                    "done": False,
                    "total_moves": 0,
                    "legal_moves": 0,
                })

            for _ in range(max_steps):
                active_indices = [i for i, rec in enumerate(records) if not rec["done"]]
                if not active_indices:
                    break

                states = [records[i]["game"]._get_state() for i in active_indices]
                actions = self.predict_actions(states, temperature=temperature)

                for rec_idx, action in zip(active_indices, actions):
                    rec = records[rec_idx]
                    game = rec["game"]
                    was_done = rec["done"]
                    valid_actions = game.get_valid_actions()
                    is_legal = action in valid_actions
                    if is_legal:
                        rec["legal_moves"] += 1
                        _, _, done, _ = game.step(action)
                    else:
                        if valid_actions:
                            _, _, done, _ = game.step(valid_actions[0])
                        else:
                            done = True
                    rec["total_moves"] += 1
                    rec["done"] = bool(done)
                    if verbose and (not was_done) and rec["done"]:
                        print(f"Snapshot {rec['game_idx']} finished with score={game.score}")

            for rec in records:
                game = rec["game"]
                total_moves = rec["total_moves"]
                scores.append(game.score)
                max_tiles.append(game.get_max_tile())
                legal_move_ratios.append(rec["legal_moves"] / total_moves if total_moves > 0 else 0)
                steps_list.append(total_moves)
                completed += 1
                if verbose and completed % 20 == 0:
                    print(f"Completed {completed}/{len(samples)} snapshots")

        results = {
            'num_games': len(samples),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'max_score': int(np.max(scores)),
            'min_score': int(np.min(scores)),
            'median_score': float(np.median(scores)),
            'mean_max_tile': float(np.mean(max_tiles)),
            'max_tile_reached': int(np.max(max_tiles)),
            'legal_move_rate': float(np.mean(legal_move_ratios)),
            'mean_steps': float(np.mean(steps_list)),
            'scores': [int(s) for s in scores],
            'max_tiles': [int(t) for t in max_tiles],
            'legal_move_ratios': [float(r) for r in legal_move_ratios],
            'steps': [int(s) for s in steps_list],
            'eval_set_path': eval_set_path,
            'bucket_distribution': bucket_counter,
        }
        return results

    def _load_eval_set(self, path: str) -> List[Dict]:
        samples: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return samples

    def _init_game_from_sample(self, sample: Dict, seed: Optional[int] = None) -> Game2048:
        game = Game2048(seed=seed)
        state_text = sample["state"]
        grid = np.array(ast.literal_eval(state_text), dtype=int)
        if grid.shape != (4, 4):
            raise ValueError(f"Invalid state shape: {grid.shape}")

        game.grid = grid
        game.score = int(sample.get("start_score", 0))
        game.game_over = game._is_game_over()
        return game

    @staticmethod
    def _derive_episode_seed(
        base_seed: Optional[int],
        index: int,
        sample: Optional[Dict] = None
    ) -> Optional[int]:
        """Derive a stable per-episode seed from global seed and sample metadata."""
        if base_seed is None:
            return None

        material = f"{int(base_seed)}:{int(index)}"
        if sample:
            material += (
                f":{sample.get('source_game_id', '')}"
                f":{sample.get('source_step', '')}"
                f":{sample.get('bucket', '')}"
            )

        digest = hashlib.blake2b(material.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % (2 ** 32)

    def print_results(self, results: Dict) -> None:
        """Print evaluation results"""
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        print(f"\nGames played: {results['num_games']}")
        print(f"\n--- Score Statistics ---")
        print(f"Mean:   {results['mean_score']:.2f}")
        print(f"Median: {results['median_score']:.2f}")
        print(f"Std:    {results['std_score']:.2f}")
        print(f"Max:    {results['max_score']}")
        print(f"Min:    {results['min_score']}")

        print(f"\n--- Tile Statistics ---")
        print(f"Mean max tile: {results['mean_max_tile']:.2f}")
        print(f"Max tile reached: {results['max_tile_reached']}")

        # Count how many games reached each milestone
        milestones = [128, 256, 512, 1024, 2048, 4096]
        print("\n--- Milestone Achievement ---")
        for milestone in milestones:
            count = sum(1 for t in results['max_tiles'] if t >= milestone)
            pct = 100 * count / results['num_games']
            print(f"{milestone:4d}: {count:3d} games ({pct:5.1f}%)")

        print(f"\n--- Move Statistics ---")
        print(f"Legal move rate: {results['legal_move_rate']*100:.1f}%")
        print(f"Mean steps: {results['mean_steps']:.1f}")

        print("\n" + "=" * 50)

    def save_results(self, results: Dict, output_path: str) -> None:
        """
        Save evaluation results to JSON.

        Args:
            results: Results dictionary
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")

    def visualize_results(self, results: Dict, output_path: str = "eval_results.png") -> None:
        """
        Visualize evaluation results.

        Args:
            results: Results dictionary
            output_path: Output file path
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Score distribution
        axes[0, 0].hist(results['scores'], bins=30, color='steelblue', edgecolor='black')
        axes[0, 0].set_title('Score Distribution', fontsize=14)
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(results['mean_score'], color='red', linestyle='--',
                          label=f"Mean: {results['mean_score']:.0f}")
        axes[0, 0].legend()

        # Max tile distribution
        axes[0, 1].hist(results['max_tiles'], bins=range(0, 2048 + 256, 256),
                       color='coral', edgecolor='black')
        axes[0, 1].set_title('Max Tile Distribution', fontsize=14)
        axes[0, 1].set_xlabel('Max Tile Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xticks([256, 512, 1024, 2048, 4096])

        # Legal move rate
        axes[1, 0].bar(['Legal Move Rate'], [results['legal_move_rate']],
                      color='lightgreen', edgecolor='black')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_title(f'Legal Move Rate: {results["legal_move_rate"]*100:.1f}%',
                            fontsize=14)
        axes[1, 0].set_ylabel('Rate')

        # Steps distribution
        axes[1, 1].hist(results['steps'], bins=30, color='plum', edgecolor='black')
        axes[1, 1].set_title('Steps per Game', fontsize=14)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(results['mean_steps'], color='red', linestyle='--',
                          label=f"Mean: {results['mean_steps']:.0f}")
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to {output_path}")
        plt.close()


def main():
    """Main entry point for evaluation"""
    parser = argparse.ArgumentParser(
        description='Evaluate 2048 game model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate base model (before training)
  python -m src.eval.evaluator --is_base_model

  # Evaluate fine-tuned model
  python -m src.eval.evaluator --model_path ./checkpoints/sft

  # 使用 vLLM 加速评估 (10-50x)
  python -m src.eval.evaluator --model_path ./checkpoints/sft --use_vllm --vllm_quantization bitsandbytes

  # 使用 4-bit 量化评估
  python -m src.eval.evaluator --model_path ./checkpoints/sft --load_in_4bit

  # Compare base model vs fine-tuned
  python -m src.eval.evaluator --is_base_model --output_dir ./eval/base
  python -m src.eval.evaluator --model_path ./checkpoints/sft --output_dir ./eval/sft
        """
    )

    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to LoRA model checkpoint (omit for base model)')
    parser.add_argument('--base_model', type=str, default="Qwen/Qwen3-1.7B",
                        help='Base model name (default: Qwen/Qwen3-1.7B)')
    parser.add_argument('--is_base_model', action='store_true',
                        help='Evaluate base model without LoRA weights')
    parser.add_argument('--num_games', type=int, default=100,
                        help='Number of games to evaluate (default: 100)')
    parser.add_argument('--max_steps', type=int, default=1000,
                        help='Maximum steps per game (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for parallel inference (default: 1)')
    parser.add_argument('--temperature', type=float, default=None,
                        help='Sampling temperature override (default: thinking=0.6, non-thinking=0.7)')
    parser.add_argument('--presence_penalty', type=float, default=0.0,
                        help='Presence penalty for supported frameworks (0~2, default: 0)')
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument('--use_thinking', dest='use_thinking', action='store_true',
                             help='Use thinking mode sampling defaults')
    think_group.add_argument('--no_thinking', dest='use_thinking', action='store_false',
                             help='Use non-thinking mode sampling defaults')
    parser.set_defaults(use_thinking=True)
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible evaluation')
    parser.add_argument('--output_dir', type=str, default="data/eval",
                        help='Output directory (default: data/eval)')
    parser.add_argument('--eval_set', type=str, default=None,
                        help='Fixed eval set path (jsonl). If set, evaluate from snapshots.')
    parser.add_argument('--no_visualize', action='store_true',
                        help='Skip visualization')

    # 优化选项
    parser.add_argument('--use_vllm', action='store_true',
                        help='使用 vLLM 加速评估 (10-50x)')
    parser.add_argument(
        '--vllm_quantization',
        type=str,
        default=None,
        help="vLLM 量化方式（如 bitsandbytes/awq/gptq；int8 会自动映射到 bitsandbytes）",
    )
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='使用 4-bit 量化')
    parser.add_argument('--load_in_8bit', action='store_true',
                        help='使用 8-bit 量化')

    args = parser.parse_args()

    # Validate arguments
    if args.is_base_model and args.model_path:
        print("⚠️  Warning: --is_base_model flag set, ignoring --model_path")
        args.model_path = None

    if not args.is_base_model and not args.model_path:
        print("❌ Error: Either --model_path or --is_base_model must be specified")
        print("   Use --is_base_model to evaluate the base model")
        print("   Use --model_path <path> to evaluate a fine-tuned model")
        parser.print_help()
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator
    evaluator = Game2048Evaluator(
        model_path=args.model_path,
        base_model=args.base_model,
        is_base_model=args.is_base_model,
        use_thinking=args.use_thinking,
        presence_penalty=args.presence_penalty,
        use_vllm=args.use_vllm,
        vllm_quantization=args.vllm_quantization,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
    )

    # Run evaluation
    if args.eval_set:
        print(f"\n🎮 Evaluating model on eval set: {args.eval_set}")
    else:
        print(f"\n🎮 Evaluating model on {args.num_games} games...")
    if args.temperature is None:
        default_temp = THINKING_SAMPLING_DEFAULTS["temperature"] if args.use_thinking else NON_THINKING_SAMPLING_DEFAULTS["temperature"]
        print(f"   Temperature: auto ({default_temp})")
    else:
        print(f"   Temperature: {args.temperature}")
    print(f"   use_thinking: {args.use_thinking}")
    print(f"   presence_penalty: {args.presence_penalty}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   Max steps: {args.max_steps}\n")

    if args.eval_set:
        results = evaluator.evaluate_from_eval_set(
            eval_set_path=args.eval_set,
            max_steps=args.max_steps,
            temperature=args.temperature,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    else:
        results = evaluator.evaluate(
            num_games=args.num_games,
            max_steps=args.max_steps,
            temperature=args.temperature,
            seed=args.seed,
            batch_size=args.batch_size,
        )

    # Add model type to results
    results['model_type'] = evaluator.model_type
    results['base_model'] = args.base_model
    results['seed'] = args.seed
    results['use_thinking'] = args.use_thinking
    results['presence_penalty'] = args.presence_penalty
    results['temperature'] = (
        args.temperature
        if args.temperature is not None
        else (THINKING_SAMPLING_DEFAULTS["temperature"] if args.use_thinking else NON_THINKING_SAMPLING_DEFAULTS["temperature"])
    )
    if args.model_path:
        results['model_path'] = args.model_path

    # Print results
    evaluator.print_results(results)

    # Save results
    results_path = output_dir / "eval_results.json"
    evaluator.save_results(results, str(results_path))

    # Visualize
    if not args.no_visualize:
        viz_path = output_dir / "eval_results.png"
        evaluator.visualize_results(results, str(viz_path))


if __name__ == "__main__":
    main()
