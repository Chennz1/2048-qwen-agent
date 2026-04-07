"""
使用TRL库进行2048游戏模型训练

核心技术栈：
- TRL (SFTTrainer, PPOTrainer)
- Transformers + PEFT
- Accelerate (混合精度/分布式)
- WandB (实验追踪)

优化支持:
- Unsloth + bitsandbytes + Flash Attention 2 (训练加速 3-5x)
- vLLM + INT8量化 (推理加速 10-50x)
"""

import os
import inspect
import types
from typing import Any, Dict, Optional
import torch

# Hugging Face mirror defaults (honor existing env if user already set it).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_from_disk
from src.envs.game_2048 import ACTION_MAP
from src.utils.action_stats import ActionWindowStats
from src.utils.monitoring import normalize_monitor_backend, report_to_list

try:
    import bitsandbytes
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    from trl import SFTConfig, SFTTrainer
    TRL_AVAILABLE = True
except ImportError:
    SFTConfig = None
    SFTTrainer = None
    TRL_AVAILABLE = False


_ACTION_CHAR_TO_ID = {v: k for k, v in ACTION_MAP.items()}


def _try_import_unsloth():
    try:
        from unsloth import FastModel
        return FastModel
    except Exception:
        return None


def _is_message_sequence(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and isinstance(value[0], dict)
        and "role" in value[0]
        and "content" in value[0]
    )


def _ensure_prompt_completion_dataset(dataset, split_name: str) -> None:
    cols = set(getattr(dataset, "column_names", []))
    required = {"prompt", "completion"}
    missing = required - cols
    if missing:
        raise ValueError(f"{split_name} dataset missing required columns: {sorted(missing)}")
    if len(dataset) == 0:
        return

    sample = dataset[0]
    if not _is_message_sequence(sample.get("prompt")):
        raise ValueError(f"{split_name} sample `prompt` must be non-empty message list")
    if not _is_message_sequence(sample.get("completion")):
        raise ValueError(f"{split_name} sample `completion` must be non-empty message list")


def _decode_action_from_token_id(tokenizer, token_id: int) -> Optional[int]:
    try:
        piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return None
    normalized = str(piece).strip().lstrip("Ġ").lstrip("▁").strip()
    if len(normalized) == 1 and normalized in _ACTION_CHAR_TO_ID:
        return int(_ACTION_CHAR_TO_ID[normalized])
    return None


def _build_action_token_map(tokenizer) -> Dict[int, int]:
    token_to_action: Dict[int, int] = {}
    for action_id, char in ACTION_MAP.items():
        try:
            token_ids = tokenizer.encode(char, add_special_tokens=False)
        except Exception:
            continue
        if len(token_ids) == 1:
            token_to_action[int(token_ids[0])] = int(action_id)
    return token_to_action


def _extract_logits_from_outputs(outputs: Any) -> Any:
    if outputs is None:
        return None
    if isinstance(outputs, dict):
        return outputs.get("logits")
    if hasattr(outputs, "get"):
        try:
            return outputs.get("logits")
        except Exception:
            pass
    if hasattr(outputs, "logits"):
        return getattr(outputs, "logits", None)
    if isinstance(outputs, (list, tuple)):
        for item in outputs:
            if hasattr(item, "shape") and len(getattr(item, "shape", [])) == 3:
                return item
    return None


def _extract_action_target_from_row(
    *,
    label_row: Any,
    valid_positions: Any,
    token_to_action: Dict[int, int],
    tokenizer: Any,
) -> Optional[tuple[int, int]]:
    """Find last action token in supervised labels: return (pos, action_id)."""
    for pos_tensor in reversed(valid_positions):
        pos = int(pos_tensor.item())
        target_token = int(label_row[pos].item())
        target_action = token_to_action.get(target_token)
        if target_action is not None:
            return pos, int(target_action)
    return None


def _update_sft_action_window_stats(
    *,
    stats: ActionWindowStats,
    labels: Any,
    logits: Any,
    token_to_action: Dict[int, int],
    tokenizer: Any,
) -> None:
    if labels is None or logits is None:
        return
    if not hasattr(labels, "shape") or not hasattr(logits, "shape"):
        return
    if len(labels.shape) != 2 or len(logits.shape) != 3:
        return
    if logits.shape[0] != labels.shape[0] or logits.shape[1] != labels.shape[1]:
        return

    with torch.no_grad():
        pred_ids = torch.argmax(logits, dim=-1)
        label_mask = labels.ne(-100)

        for row_idx in range(int(labels.shape[0])):
            valid_positions = torch.nonzero(label_mask[row_idx], as_tuple=False)
            if valid_positions.numel() == 0:
                continue

            action_target = _extract_action_target_from_row(
                label_row=labels[row_idx],
                valid_positions=valid_positions,
                token_to_action=token_to_action,
                tokenizer=tokenizer,
            )
            if action_target is None:
                continue
            action_pos, target_action = action_target
            # CausalLM next-token alignment:
            # logits[t] predicts labels[t+1], so action label at p should use logits[p-1].
            if action_pos <= 0:
                continue
            pred_token = int(pred_ids[row_idx, action_pos - 1].item())

            pred_action = token_to_action.get(pred_token)
            if pred_action is None:
                pred_action = _decode_action_from_token_id(tokenizer, pred_token)

            parsed = pred_action is not None
            stats.update(
                parsed=parsed,
                legal=parsed,
                correct=bool(parsed and pred_action == target_action),
            )


def _patch_sft_trainer_action_window_logging(trainer: Any, tokenizer: Any) -> ActionWindowStats:
    """Patch SFT trainer to log windowed action stats on default log interval."""
    stats = ActionWindowStats()
    token_to_action = _build_action_token_map(tokenizer)
    if not token_to_action:
        print("[SFT] 警告: 未找到单token动作映射，action_acc 可能偏低。")

    original_compute_loss = trainer.compute_loss
    compute_sig = inspect.signature(original_compute_loss)
    supports_return_outputs = "return_outputs" in compute_sig.parameters

    def _wrapped_compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        outputs = None
        if supports_return_outputs:
            loss, outputs = original_compute_loss(
                model,
                inputs,
                return_outputs=True,
                *args,
                **kwargs,
            )
        else:
            loss = original_compute_loss(model, inputs, *args, **kwargs)

        try:
            labels = inputs.get("labels") if hasattr(inputs, "get") else None
            _update_sft_action_window_stats(
                stats=stats,
                labels=labels,
                logits=_extract_logits_from_outputs(outputs),
                token_to_action=token_to_action,
                tokenizer=tokenizer,
            )
        except Exception:
            pass

        if return_outputs:
            return loss, outputs
        return loss

    original_log = trainer.log

    def _wrapped_log(self, logs, *args, **kwargs):
        merged = dict(logs) if isinstance(logs, dict) else {}
        merged.update(stats.flush())
        return original_log(merged, *args, **kwargs)

    trainer.compute_loss = types.MethodType(_wrapped_compute_loss, trainer)
    trainer.log = types.MethodType(_wrapped_log, trainer)
    return stats


def _build_sft_config(
    *,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    monitor_backend: str,
    run_name: str,
    use_unsloth: bool,
    load_in_4bit: bool,
    load_in_8bit: bool,
    tokenizer=None,
    completion_only_loss: bool = True,
    disable_tqdm: bool = False,
):
    """Build SFTConfig with runtime compatibility across TRL versions."""
    params = set(inspect.signature(SFTConfig.__init__).parameters.keys())
    # Thinking samples are typically prompt(500+) + completion(600+).
    # Keep non-Unsloth SFT at 1536 to avoid truncating most samples.
    seq_len = 2048 if use_unsloth else 1536
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "learning_rate": learning_rate,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "disable_tqdm": bool(disable_tqdm),
        "logging_steps": 4,
        "save_strategy": "epoch",
        "save_total_limit": 3,
        "gradient_checkpointing": True,
        "report_to": report_to_list(monitor_backend),
        "run_name": run_name,
    }
    # Keep fp16/bf16 mutually exclusive to avoid AMP GradScaler errors.
    if "bf16" in params:
        kwargs["bf16"] = use_bf16
    if "fp16" in params:
        if "bf16" in params:
            kwargs["fp16"] = not use_bf16
        else:
            # Older configs without bf16 field: on bf16-capable GPUs, avoid fp16
            # to prevent GradScaler trying to unscale bf16 gradients.
            kwargs["fp16"] = not use_bf16

    # Disable legacy validation-loss evaluation path.
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "no"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "no"

    if "dataset_text_field" in params:
        # We train with prompt/completion conversational samples, not legacy text field.
        kwargs["dataset_text_field"] = None
    if "completion_only_loss" in params:
        kwargs["completion_only_loss"] = bool(completion_only_loss)

    if "max_seq_length" in params:
        kwargs["max_seq_length"] = seq_len
    elif "max_length" in params:
        kwargs["max_length"] = seq_len

    # Newer TRL versions may default to placeholder tokens (e.g. <EOS_TOKEN>).
    # Always bind tokenizer-native special tokens when supported.
    if "eos_token" in params and tokenizer is not None and getattr(tokenizer, "eos_token", None):
        kwargs["eos_token"] = tokenizer.eos_token
    if "pad_token" in params and tokenizer is not None and getattr(tokenizer, "pad_token", None):
        kwargs["pad_token"] = tokenizer.pad_token
    if "eos_token_id" in params and tokenizer is not None and getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = int(tokenizer.eos_token_id)
    if "pad_token_id" in params and tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = int(tokenizer.pad_token_id)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    cfg = SFTConfig(**filtered)

    # Defensive precision overwrite in case ctor defaults/env re-enable fp16.
    if hasattr(cfg, "bf16"):
        setattr(cfg, "bf16", bool(use_bf16))
    if hasattr(cfg, "fp16"):
        if hasattr(cfg, "bf16"):
            setattr(cfg, "fp16", not bool(use_bf16))
        else:
            setattr(cfg, "fp16", not bool(use_bf16))

    # Defensive overwrite for newer TRL versions that may keep placeholder defaults
    # like "<EOS_TOKEN>" even when ctor kwargs differ across versions.
    if tokenizer is not None:
        if hasattr(cfg, "eos_token") and getattr(tokenizer, "eos_token", None):
            setattr(cfg, "eos_token", tokenizer.eos_token)
        if hasattr(cfg, "pad_token") and getattr(tokenizer, "pad_token", None):
            setattr(cfg, "pad_token", tokenizer.pad_token)
        if hasattr(cfg, "eos_token_id") and getattr(tokenizer, "eos_token_id", None) is not None:
            setattr(cfg, "eos_token_id", int(tokenizer.eos_token_id))
        if hasattr(cfg, "pad_token_id") and getattr(tokenizer, "pad_token_id", None) is not None:
            setattr(cfg, "pad_token_id", int(tokenizer.pad_token_id))

    return cfg


def _build_sft_trainer(
    *,
    model,
    sft_config,
    train_dataset,
    tokenizer,
    data_collator=None,
):
    """Build SFTTrainer with runtime compatibility across TRL versions."""
    params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
    }

    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "eos_token" in params and getattr(tokenizer, "eos_token", None):
        kwargs["eos_token"] = tokenizer.eos_token
    if "pad_token" in params and getattr(tokenizer, "pad_token", None):
        kwargs["pad_token"] = tokenizer.pad_token
    if "eos_token_id" in params and getattr(tokenizer, "eos_token_id", None) is not None:
        kwargs["eos_token_id"] = int(tokenizer.eos_token_id)
    if "pad_token_id" in params and getattr(tokenizer, "pad_token_id", None) is not None:
        kwargs["pad_token_id"] = int(tokenizer.pad_token_id)
    if "data_collator" in params and data_collator is not None:
        kwargs["data_collator"] = data_collator
    return SFTTrainer(**kwargs)


def train_sft(
    model_name: str = "Qwen/Qwen3-1.7B",
    train_data: str = "data/processed/train",
    val_data: str = "data/processed/val",
    output_dir: str = "./checkpoints/sft",
    num_epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation: int = 4,
    learning_rate: float = 5e-5,
    use_wandb: bool = True,
    monitor_backend: str = "wandb",
    # 优化选项 (默认开启)
    use_unsloth: bool = True,
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    use_flash_attn: bool = False,
    completion_only_loss: bool = True,
    disable_tqdm: bool = False,
):
    """
    使用TRL的SFTTrainer进行监督微调

    这是学习TRL的第一步 - SFT是最常用的技术

    Args:
        model_name: 基座模型名称
        train_data: 训练数据路径
        val_data: 验证数据路径
        output_dir: 输出目录
        num_epochs: 训练轮数
        batch_size: 批次大小
        gradient_accumulation: 梯度累积步数
        learning_rate: 学习率
        use_wandb: 兼容旧参数，等价于 monitor_backend=wandb/none
        monitor_backend: 监控后端 (wandb/tensorboard/none)
        use_unsloth: 是否使用Unsloth优化
        load_in_4bit: 是否使用4-bit量化
        load_in_8bit: 是否使用8-bit量化
        use_flash_attn: 是否使用Flash Attention 2
    """
    if not TRL_AVAILABLE:
        raise RuntimeError("`trl` is required for SFT training. Please install `trl>=0.12.0`.")

    monitor_backend = normalize_monitor_backend(
        monitor_backend,
        use_wandb=use_wandb,
    )
    fast_model_cls = _try_import_unsloth()
    unsloth_available = fast_model_cls is not None

    # 显示优化配置
    print("\n" + "=" * 70)
    print("⚡ 优化配置")
    print("=" * 70)
    print(f"📈 监控后端: {monitor_backend}")
    use_optimized = use_unsloth or load_in_4bit or load_in_8bit or use_flash_attn

    # 检查flash-attn是否可用
    try:
        import flash_attn
        FLASH_ATTN_AVAILABLE = True
    except ImportError:
        FLASH_ATTN_AVAILABLE = False

    if use_unsloth:
        if unsloth_available:
            if not FLASH_ATTN_AVAILABLE:
                print("⚠️ Unsloth需要flash-attn，回退到标准模式")
                print("   提示：安装flash-attn可获得更好性能（可选）")
                use_unsloth = False
            else:
                print("✅ Unsloth: 启用 (3-5x 加速)")
        else:
            print("⚠️ Unsloth: 未安装，回退到标准模式")
            use_unsloth = False

    if load_in_4bit:
        if BNB_AVAILABLE:
            print("✅ 4-bit量化: 启用 (显存 ~3GB)")
        else:
            print("⚠️ bitsandbytes未安装，无法使用4-bit量化")
            load_in_4bit = False

    if load_in_8bit:
        if BNB_AVAILABLE:
            print("✅ 8-bit量化: 启用")
        else:
            print("⚠️ bitsandbytes未安装，无法使用8-bit量化")
            load_in_8bit = False

    if use_flash_attn and FLASH_ATTN_AVAILABLE:
        print("✅ Flash Attention 2: 启用")

    if not use_optimized:
        print("ℹ️ 使用标准训练模式")

    print("=" * 70)

    # 1. 加载数据集
    print("\n📊 加载数据集...")
    train_dataset = load_from_disk(train_data)

    print(f"  训练集: {len(train_dataset)} 样本")
    _ensure_prompt_completion_dataset(train_dataset, "train")
    print("  数据集格式: conversational prompt-completion")

    # 2. 加载模型和Tokenizer
    print("\n🔤 加载Tokenizer...")

    # 使用Unsloth（只有当flash-attn可用时）
    if use_unsloth and unsloth_available:
        print("📦 使用Unsloth加载模型...")
        model, tokenizer = fast_model_cls.from_pretrained(
            model_name=model_name,
            load_in_4bit=load_in_4bit,
            max_seq_length=2048,
            device_map="auto",
        )
    else:
        # 标准加载
        print("📦 加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # 配置量化
        bnb_config = None
        if load_in_4bit and BNB_AVAILABLE:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif load_in_8bit and BNB_AVAILABLE:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # 加载模型
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
        }

        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    # 3. 配置LoRA
    print("🔧 配置LoRA...")
    from src.models.lora_config import get_lora_config

    lora_config = get_lora_config(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05
    )

    # 量化模型必须在Trainer初始化前显式挂载可训练LoRA，避免“pure quantized model”报错。
    from peft import get_peft_model, prepare_model_for_kbit_training
    if load_in_4bit or load_in_8bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # 4. 配置SFT训练（按当前TRL版本动态适配参数）
    sft_config = _build_sft_config(
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        monitor_backend=monitor_backend,
        run_name=f"2048-sft-{model_name.split('/')[-1]}",
        use_unsloth=use_unsloth,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        tokenizer=tokenizer,
        completion_only_loss=completion_only_loss,
        disable_tqdm=disable_tqdm,
    )

    print(
        "🧪 Precision config:",
        f"fp16={getattr(sft_config, 'fp16', None)}",
        f"bf16={getattr(sft_config, 'bf16', None)}",
    )

    if hasattr(sft_config, "completion_only_loss"):
        setattr(sft_config, "completion_only_loss", True)
    else:
        raise RuntimeError(
            "当前TRL版本不支持 `completion_only_loss`。请升级TRL后再训练（建议 >= 0.15）。"
        )
    print("🧠 使用 TRL completion_only_loss=True（官方路径）")

    # 5. 创建Trainer
    print("🚀 创建SFTTrainer...")
    trainer = _build_sft_trainer(
        model=model,
        sft_config=sft_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    action_stats = _patch_sft_trainer_action_window_logging(trainer, tokenizer)

    # 显示可训练参数
    trainable_params = trainer.model.get_nb_trainable_parameters()
    print(f"\n📈 可训练参数: {trainable_params}\n")

    # 6. 训练
    print("开始训练...")
    trainer.train()
    tail_metrics = action_stats.flush()
    if tail_metrics:
        trainer.log(tail_metrics)

    # 7. 保存模型
    print(f"\n💾 保存模型到 {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    return trainer


def train_with_custom_reward(
    model_path: str,
    output_dir: str = "./checkpoints/rl",
    num_episodes: int = 1000,
    use_wandb: bool = True,
):
    """
    使用自定义奖励函数进行强化学习训练

    展示如何将TRL与环境集成

    Args:
        model_path: SFT训练后的模型路径
        output_dir: 输出目录
        num_episodes: 训练回合数
        use_wandb: 是否使用WandB
    """
    from trl import PPOTrainer, PPOConfig
    from transformers import AutoModelForCausalLM
    from src.envs.game_2048 import Game2048
    import wandb

    # 初始化WandB
    if use_wandb:
        wandb.init(
            project="2048-game-rl",
            name="ppo-training",
            config={
                "model": model_path,
                "num_episodes": num_episodes,
            }
        )

    print("🎮 加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # PPO配置
    ppo_config = PPOConfig(
        learning_rate=1e-5,
        batch_size=128,
        mini_batch_size=32,
        gradient_accumulation_steps=4,
    )
    _ = ppo_config

    print("创建PPOTrainer...")
    # 注意：这里需要自定义reward函数
    # PPOTrainer主要用于对话，游戏需要自定义

    print("\n💡 提示：对于游戏任务，建议使用自定义RL循环")
    print("   可以参考 TRL 的PPOTrainer实现，但需要适配游戏环境")
    print("   或者使用 ReST (迭代式自训练) 方法")

    if use_wandb:
        wandb.finish()


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="使用TRL训练2048游戏模型")
    parser.add_argument("--mode", type=str, default="sft",
                        choices=["sft", "rl"],
                        help="训练模式: sft=监督微调, rl=强化学习")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B",
                        help="基座模型")
    parser.add_argument("--base_model", type=str, default=None,
                        help="基座模型（--model 的别名）")
    parser.add_argument("--train_data", type=str, default="data/processed/train")
    parser.add_argument("--val_data", type=str, default="data/processed/val")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="梯度累计步数")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--no_wandb", action="store_true",
                        help="不使用WandB")
    parser.add_argument(
        "--monitor_backend",
        type=str,
        default=None,
        choices=["wandb", "tensorboard", "none"],
        help="监控后端 (wandb/tensorboard/none)",
    )
    parser.add_argument("--disable_tqdm", action="store_true")

    # 优化选项
    parser.add_argument("--use_unsloth", action="store_true",
                        help="使用Unsloth优化 (3-5x加速)")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="使用4-bit量化 (显存~3GB)")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="使用8-bit量化")
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="使用Flash Attention 2")

    args = parser.parse_args()

    model_name = args.base_model or args.model

    monitor_backend = normalize_monitor_backend(
        args.monitor_backend,
        no_wandb=args.no_wandb,
    )

    if args.mode == "sft":
        print("=" * 60)
        print("监督微调 (SFT)")
        print("=" * 60)

        train_sft(
            model_name=model_name,
            train_data=args.train_data,
            val_data=args.val_data,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.grad_accum,
            learning_rate=args.lr,
            use_wandb=not args.no_wandb,
            monitor_backend=monitor_backend,
            use_unsloth=args.use_unsloth,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            use_flash_attn=args.use_flash_attn,
            disable_tqdm=args.disable_tqdm,
        )

        print("\n✅ SFT训练完成!")
        print(f"📍 模型保存在: {args.output_dir}")

    elif args.mode == "rl":
        print("=" * 60)
        print("强化学习训练 (RL)")
        print("=" * 60)
        print("\n💡 建议：先完成SFT训练，再进行RL微调")
        print("   RL模式需要SFT的模型作为起点")


if __name__ == "__main__":
    main()
