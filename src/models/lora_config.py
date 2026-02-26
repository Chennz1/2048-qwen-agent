"""
LoRA Configuration and Model Setup
Configures LoRA for fine-tuning LLMs on 2048 game data.
"""

import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional


def get_lora_config(
    r: int = 32,
    lora_alpha: int = 64,
    target_modules: Optional[list] = None,
    lora_dropout: float = 0.05,
    bias: str = "none"
) -> LoraConfig:
    """
    Get LoRA configuration.

    Args:
        r: LoRA rank (higher = more parameters, better potential performance)
        lora_alpha: LoRA scaling factor
        target_modules: List of modules to apply LoRA to
        lora_dropout: Dropout probability
        bias: Bias setting ("none", "all", "lora_only")

    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default target modules for Qwen/Llama models
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType.CAUSAL_LM
    )


def setup_model(
    base_model_name: str = "Qwen/Qwen3-1.7B",
    lora_config: Optional[LoraConfig] = None,
    torch_dtype: torch.dtype = torch.float16,
    device_map: str = "auto"
):
    """
    Set up model and tokenizer with LoRA.

    Args:
        base_model_name: HuggingFace model name or local path
        lora_config: LoRA configuration (uses default if None)
        torch_dtype: Data type for model weights
        device_map: Device mapping strategy

    Returns:
        (model, tokenizer)
    """
    print(f"Loading base model: {base_model_name}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True
    )

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Create or use provided LoRA config
    if lora_config is None:
        lora_config = get_lora_config()

    # Apply LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    trainable_params = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable_params}")

    return model, tokenizer


def print_model_info(model, tokenizer) -> None:
    """Print model information"""
    print("\n=== Model Info ===")
    print(f"Model type: {type(model).__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    print(f"Eos token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")


# Preset configurations for different models

QWEN_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05
}

LLAMA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05
}

GEMMA_CONFIG = {
    "r": 32,
    "lora_alpha": 64,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05
}


def get_preset_config(model_name: str) -> dict:
    """
    Get preset LoRA config for a model.

    Args:
        model_name: Model name (e.g., "qwen", "llama", "gemma")

    Returns:
        Configuration dictionary
    """
    model_name_lower = model_name.lower()

    if "qwen" in model_name_lower:
        return QWEN_CONFIG
    elif "llama" in model_name_lower:
        return LLAMA_CONFIG
    elif "gemma" in model_name_lower:
        return GEMMA_CONFIG
    else:
        # Default config
        return QWEN_CONFIG


if __name__ == "__main__":
    # Test loading model
    print("Testing model setup...")

    model, tokenizer = setup_model(
        base_model_name="Qwen/Qwen3-1.7B-Instruct"
    )

    print_model_info(model, tokenizer)

    # Test tokenization
    test_text = "当前棋盘状态(4x4数组，0表示空位):\n[[2, 4, 8, 16],\n [4, 8, 16, 32],\n [0, 0, 0, 0],\n [0, 0, 0, 0]]\n\n请选择最佳着法:"
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"\nTest text length: {len(tokens['input_ids'][0])} tokens")
