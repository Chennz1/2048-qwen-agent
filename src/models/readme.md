# models 模块说明

`src/models` 包含三条训练路线：SFT、ReST、GRPO。  
目标是复用同一数据契约与 prompt 格式，避免多套训练范式分叉。

## 模块职责
- 监督微调（SFT）
- 迭代式自训练（ReST）
- 基于 TRL 的 GRPO 强化学习训练
- 提供 LoRA 配置封装

## 文件结构与作用

### `src/models/trl_train.py`（SFT 主入口）
- 关键函数：`train_sft(...)`
- 主要流程：
  - 读取 processed 训练/验证集
  - 加载 tokenizer 和模型（可选量化/加速）
  - 配置 LoRA 与 `SFTTrainer`
  - 执行训练并保存 checkpoint
- 特点：
  - 支持 `monitor_backend`（wandb/tensorboard/none）
  - 支持 `unsloth`, `4bit/8bit`, `flash-attn`

### `src/models/modern_rl_trainer.py`（ReST）
- 核心对象：
  - `ReSTConfig`：迭代训练配置
  - `GamePlayer`：使用模型进行对局采样
  - `ReSTTrainer`：完整迭代流程
- 主循环：
  1. 当前模型生成对局数据
  2. 按分数筛选高质量对局
  3. 转换成 SFT 样本
  4. 微调得到下一轮模型
  5. 评估并记录历史

### `src/models/grpo.py`（TRL-native GRPO）
- 核心对象：
  - `GRPO2048DatasetBuilder`：构建 prompt-only 数据
  - `GRPO2048Rewards`：奖励函数集合
  - `TRLGRPO2048Trainer`：封装 TRL `GRPOTrainer`
- 奖励函数：
  - `json_focused`（默认）：固定分值评估 JSON 结构、局面事实、合法性判断与动作合法性，并对 expert 最优动作命中额外加分
  - 非 `<think>` 区 JSON 非法时直接给大额负分
  - 奖励配置：`JsonRewardConfig`（无权重映射项）

### `src/models/lora_config.py`
- 作用：集中管理 LoRA 参数，减少重复配置。

## 训练数据要求
- 推荐输入：`src/data_gen/processor.py` 产出的 `data/processed/*`
- 样本字段：至少包含 `text`
- 训练模板：应与推理模板保持一致（同 tokenizer/chat_template）

## 典型入口

### SFT
```bash
python -m src.models.trl_train \
  --mode sft \
  --train_data data/processed/train \
  --val_data data/processed/val
```

### ReST
```bash
python -m src.models.modern_rl_trainer \
  --base_model ./checkpoints/sft \
  --num_iterations 5
```

### GRPO
```bash
python -m src.models.grpo \
  --model ./checkpoints/sft \
  --data_source raw \
  --input_dir data/raw
```

## 依赖与可选优化
- 必需：`transformers`, `trl`, `datasets`, `peft`
- 可选：
  - `unsloth`：训练加速
  - `bitsandbytes`：4bit/8bit 量化
  - `flash-attn`：注意力加速

## 常见问题

### 1) 训练时报 `trl` 相关错误
- 确认安装了兼容版本的 `trl`。
- 若仅做数据链路验证，可先不跑训练入口。

### 2) 显存不足
- 优先启用 `load_in_4bit`，并降低 batch size。
- 增加 `gradient_accumulation` 代替直接增大 batch。

### 3) 模型输出格式不稳定
- 检查训练与推理是否使用统一 prompt 构造（`src/data_gen/prompting.py`）。
- 检查训练样本末尾是否稳定为合法动作字符。
