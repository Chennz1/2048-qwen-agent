"""TRL-native GRPO trainer for 2048 next-board prediction with optional LoRA."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

from src.models.grpo import (
    _build_grpo_config,
    build_grpo_generation_kwargs,
    _load_trl_grpo_symbols,
    _patch_grpo_trainer_sampler_compat,
    _resolve_grpo_model_input,
    _validate_batch_generation_compatibility,
    resolve_save_steps,
)
from src.models.lora_config import get_lora_config, get_preset_config
from src.models.next_board_grpo import (
    NextBoardGRPODatasetBuilder,
    NextBoardGRPORewards,
    NextBoardRewardConfig,
    _patch_next_board_logging,
)
from src.utils.action_stats import ActionWindowStats
from src.utils.monitoring import normalize_monitor_backend, report_to_list


os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def _load_model_with_optional_lora(
    model_name_or_path: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> Any:
    if not use_lora:
        return _resolve_grpo_model_input(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    preset = get_preset_config(model_name_or_path)
    lora_config = get_lora_config(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=preset.get("target_modules"),
        lora_dropout=float(lora_dropout),
    )

    from peft import get_peft_model

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


class TRLGRPONextBoardLoraTrainer:
    """High-level trainer using TRL GRPOTrainer for next-board prediction."""

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "./checkpoints/grpo_next_board_lora",
        use_wandb: bool = False,
        monitor_backend: str = "none",
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.monitor_backend = normalize_monitor_backend(
            monitor_backend,
            use_wandb=use_wandb,
        )
        self.use_lora = use_lora
        self.lora_r = int(lora_r)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)

    def train(
        self,
        dataset: Dataset,
        tokenizer: Optional[Any] = None,
        learning_rate: float = 1e-6,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        clip_eps: float = 0.28,
        kl_beta: float = 0.0,
        num_train_epochs: int = 1,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_generations: int = 2,
        max_prompt_length: int = 512,
        max_completion_length: int = 256,
        save_steps: float = 200,
        logging_steps: int = 4,
        disable_tqdm: bool = False,
        log_completions: bool = False,
        num_completions_to_print: int = 4,
        log_unique_prompts: bool = False,
    ) -> Any:
        GRPOConfig, GRPOTrainer = _load_trl_grpo_symbols()
        report_to = report_to_list(self.monitor_backend)
        generation_kwargs = build_grpo_generation_kwargs(tokenizer)
        resolved_save_steps = resolve_save_steps(
            save_steps=save_steps,
            dataset_size=len(dataset),
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        cfg = _build_grpo_config(
            GRPOConfig=GRPOConfig,
            output_dir=self.output_dir,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            num_train_epochs=num_train_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            save_steps=resolved_save_steps,
            logging_steps=logging_steps,
            disable_tqdm=disable_tqdm,
            generation_kwargs=generation_kwargs,
            log_completions=log_completions,
            num_completions_to_print=num_completions_to_print,
            log_unique_prompts=log_unique_prompts,
            report_to=report_to,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        )

        model_input = _load_model_with_optional_lora(
            model_name_or_path=self.model_name_or_path,
            use_lora=self.use_lora,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
        )

        reward_funcs = NextBoardGRPORewards.get_reward_funcs()
        trainer_params = set(inspect.signature(GRPOTrainer.__init__).parameters.keys())
        trainer_kwargs: Dict[str, Any] = {
            "model": model_input,
            "reward_funcs": reward_funcs,
            "args": cfg,
            "train_dataset": dataset,
        }
        if tokenizer is not None:
            if "processing_class" in trainer_params:
                trainer_kwargs["processing_class"] = tokenizer
            elif "tokenizer" in trainer_params:
                trainer_kwargs["tokenizer"] = tokenizer

        trainer = GRPOTrainer(**trainer_kwargs)
        _patch_grpo_trainer_sampler_compat(trainer)

        stats = ActionWindowStats()
        NextBoardGRPORewards.attach_window_stats(stats)
        _patch_next_board_logging(trainer, stats)

        try:
            trainer.train()
            tail_metrics = stats.flush()
            if tail_metrics:
                trainer.log(tail_metrics)
        finally:
            NextBoardGRPORewards.attach_window_stats(None)

        trainer.save_model(self.output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(self.output_dir)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "trainer": "trl_grpo_next_board_lora",
                    "model_name_or_path": self.model_name_or_path,
                    "num_train_samples": len(dataset),
                    "reward_mode": "next_board_accuracy",
                    "reward_config": vars(NextBoardGRPORewards._cfg),
                    "use_lora": bool(self.use_lora),
                    "lora_r": self.lora_r,
                    "lora_alpha": self.lora_alpha,
                    "lora_dropout": self.lora_dropout,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRL GRPO training for 2048 next-board prediction with optional LoRA"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Instruct")
    parser.add_argument("--base_model", type=str, default=None, help="Alias of --model")
    parser.add_argument("--input_dir", type=str, default="data/processed_next_board/train")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_next_board_lora")
    parser.add_argument("--num_samples", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--clip_eps", type=float, default=0.28)
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--save_steps", type=float, default=500)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--num_completions_to_print", type=int, default=4)
    parser.add_argument("--log_unique_prompts", action="store_true")
    parser.add_argument("--reward_json_invalid_penalty", type=float, default=-20.0)
    parser.add_argument("--reward_exact_board_bonus", type=float, default=6.0)
    parser.add_argument("--reward_consistent_with_env_bonus", type=float, default=2.0)
    parser.add_argument("--reward_inconsistent_with_env_penalty", type=float, default=-2.0)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--monitor_backend",
        type=str,
        default=None,
        choices=["wandb", "tensorboard", "none"],
    )
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--prefill_reasoning_prefix", action="store_true")
    args = parser.parse_args()

    model_name_or_path = args.base_model or args.model
    monitor_backend = normalize_monitor_backend(
        args.monitor_backend,
        no_wandb=args.no_wandb,
    )
    _validate_batch_generation_compatibility(
        batch_size=args.batch_size,
        num_generations=args.num_generations,
    )

    builder = NextBoardGRPODatasetBuilder(
        model_name=model_name_or_path,
        use_thinking=not args.no_thinking,
        prefill_reasoning_prefix=args.prefill_reasoning_prefix,
    )
    train_dataset = builder.build(
        data_dir=args.input_dir,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    NextBoardGRPORewards.configure(
        NextBoardRewardConfig(
            json_invalid_penalty=args.reward_json_invalid_penalty,
            exact_board_bonus=args.reward_exact_board_bonus,
            consistent_with_env_bonus=args.reward_consistent_with_env_bonus,
            inconsistent_with_env_penalty=args.reward_inconsistent_with_env_penalty,
        )
    )

    trainer = TRLGRPONextBoardLoraTrainer(
        model_name_or_path=model_name_or_path,
        output_dir=args.output_dir,
        use_wandb=not args.no_wandb,
        monitor_backend=monitor_backend,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    trainer.train(
        dataset=train_dataset,
        tokenizer=builder.tokenizer,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        disable_tqdm=args.disable_tqdm,
        log_completions=args.log_completions,
        num_completions_to_print=args.num_completions_to_print,
        log_unique_prompts=args.log_unique_prompts,
    )


if __name__ == "__main__":
    main()
