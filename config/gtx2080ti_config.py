# GTX 2080 Ti (11GB) 训练配置
# 针对 11GB 显存优化的训练参数

# ==============================================================================
# 模型加载配置 - 8-bit 量化（必需）
# ==============================================================================
MODEL_LOADING_CONFIG = {
    "model_name": "Qwen/Qwen3-1.7B",
    "load_in_8bit": True,           # ⚠️ 必需！11GB 显存无法加载 fp16 模型
    "device_map": "auto",
    "trust_remote_code": True,
    "torch_dtype": "float16",      # 仍然使用 fp16 进行计算
}

# ==============================================================================
# SFT 训练配置 - GTX 2080 Ti 优化
# ==============================================================================
SFT_TRAINING_CONFIG = {
    # 基础设置
    "output_dir": "./checkpoints/sft",
    "num_train_epochs": 3,

    # ⚠️ GTX 2080 Ti 关键配置
    "per_device_train_batch_size": 2,      # 小 batch size (2-4)
    "per_device_eval_batch_size": 4,        # 评估时可以稍大
    "gradient_accumulation_steps": 8,       # 累积 8 步 = 等效 batch_size=16

    # 优化器
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",

    # ⚠️ GTX 2080 Ti 显存优化
    "fp16": True,                          # 混合精度训练
    "gradient_checkpointing": True,         # 梯度检查点（节省显存）
    "max_seq_length": 256,                  # 减少序列长度（从 512）
    "dataloader_num_workers": 2,            # 数据加载线程数
    "dataloader_pin_memory": True,          # 加快数据传输

    # 保存和评估
    "logging_steps": 20,
    "save_strategy": "epoch",
    "eval_strategy": "epoch",
    "save_total_limit": 2,

    # 性能优化
    "optim": "adamw_torch",                 # 使用 PyTorch 原生 AdamW（更省显存）
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,                   # 梯度裁剪
}

# ==============================================================================
# LoRA 配置 - GTX 2080 Ti 优化
# ==============================================================================
LORA_CONFIG = {
    # LoRA rank (影响参数量和显存)
    "r": 16,                                # 降低 rank (从 32 到 16)，节省显存
    "lora_alpha": 32,                        # alpha = 2*r
    "target_modules": [                      # 关键模块
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    # 注意：GTX 2080 Ti 11GB 可能无法包含 gate_proj, up_proj, down_proj
    # 如果显存不足，注释掉下面三行
    # "gate_proj",
    # "up_proj",
    # "down_proj"

    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# ==============================================================================
# ReST 训练配置 - GTX 2080 Ti 优化
# ==============================================================================
REST_TRAINING_CONFIG = {
    # 迭代配置
    "num_iterations": 5,
    "games_per_iteration": 500,             # 减少游戏数（从 1000）

    # SFT 配置
    "sft_epochs": 1,
    "learning_rate": 5e-5,
    "batch_size": 2,                        # 小 batch size
    "gradient_accumulation": 4,             # 梯度累积

    # 数据筛选
    "top_k_ratio": 0.5,
    "min_score_threshold": 500,

    # 评估
    "eval_games": 50,                       # 减少评估游戏数

    # 生成配置
    "temperature": 0.5,
}

# ==============================================================================
# 评估配置 - GTX 2080 Ti
# ==============================================================================
EVAL_CONFIG = {
    "num_games": 50,                         # 减少评估游戏数
    "max_steps": 1000,
    "temperature": 0.1,
    "batch_size": 1,                         # 评估时用 batch size 1
}

# ==============================================================================
# 显存监控脚本
# ==============================================================================
MEMORY_MONITOR_SCRIPT = """
# 实时监控 GPU 显存使用
watch -n 1 nvidia-smi

# 或使用 Python 监控
python -c "
import time
import nvidia_ml_py3 as nvml

nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)

while True:
    info = nvml.nvmlDeviceGetMemoryInfo(handle)
    used = info.used / 1024**3
    total = info.total / 1024**3
    print(f'GPU 显存: {used:.2f}GB / {total:.2f}GB ({used/total*100:.1f}%)')
    time.sleep(5)
"
"""

# ==============================================================================
# 显存使用估算（Qwen3-1.7B + LoRA）
# ==============================================================================
MEMORY_ESTIMATES = {
    "模型加载 (8-bit)": "3.5 GB",
    "LoRA 权重": "0.2 GB",
    "优化器状态": "1.5 GB",
    "梯度 (fp16)": "0.5 GB",
    "激活值 (batch_size=2, seq_len=256)": "2.0 GB",
    "总计": "约 7.7 GB",
    "剩余空间": "约 3.3 GB",
    "安全边际": "✅ 充足",
}

# ==============================================================================
# 如果遇到 OOM (Out of Memory) 的解决方案
# ==============================================================================
OOM_SOLUTIONS = [
    "1. 减小 batch_size (从 2 降到 1)",
    "2. 增加 gradient_accumulation_steps (从 8 增到 16)",
    "3. 减少 max_seq_length (从 256 降到 128)",
    "4. 使用 4-bit 量化 (load_in_4bit=True)",
    "5. 减少 LoRA rank (从 16 降到 8)",
    "6. 移除 gate_proj, up_proj, down_proj",
    "7. 使用更小的模型 (Qwen3-0.5B)",
]

# ==============================================================================
# 性能预期（GTX 2080 Ti）
# ==============================================================================
PERFORMANCE_EXPECTATIONS = {
    "SFT 训练速度": "约 2-3 小时/epoch",
    "总训练时间 (3 epochs)": "约 6-9 小时",
    "ReST 训练时间": "约 8-12 小时",
    "推理速度": "约 15-30 ms/token",
    "预期分数 (SFT)": "1800-2500",
    "预期分数 (ReST)": "3000-4500",
}

# ==============================================================================
# 快速测试命令
# ==============================================================================
QUICK_TEST_COMMANDS = """
# 1. 测试模型加载（8-bit）
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-1.7B',
    load_in_8bit=True,
    device_map='auto'
)
print(f'模型加载成功！显存: {torch.cuda.memory_allocated()/1024**3:.2f}GB')
"

# 2. 测试训练循环（1步）
python -m src.models.trl_train \\
    --model Qwen/Qwen3-1.7B \\
    --train_data data/processed/train \\
    --epochs 1 \\
    --batch_size 2 \\
    --output_dir ./test_run

# 3. 监控显存
watch -n 1 nvidia-smi
"""

if __name__ == "__main__":
    print("="*60)
    print("GTX 2080 Ti (11GB) - 训练配置")
    print("="*60)
    print()
    print("推荐训练配置：")
    print(f"  batch_size: {SFT_TRAINING_CONFIG['per_device_train_batch_size']}")
    print(f"  gradient_accumulation: {SFT_TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"  fp16: {SFT_TRAINING_CONFIG['fp16']}")
    print(f"  gradient_checkpointing: {SFT_TRAINING_CONFIG['gradient_checkpointing']}")
    print(f"  max_seq_length: {SFT_TRAINING_CONFIG['max_seq_length']}")
    print(f"  LoRA r: {LORA_CONFIG['r']}")
    print()
    print("预期显存使用:")
    for key, value in MEMORY_ESTIMATES.items():
        print(f"  {key}: {value}")
    print()
    print("性能预期:")
    for key, value in PERFORMANCE_EXPECTATIONS.items():
        print(f"  {key}: {value}")
    print("="*60)
