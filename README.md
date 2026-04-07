# 2048 + Qwen3 Training Playground

这是一个面向 2048 决策任务的 LLM 训练实验项目，目标是把“数据生成、数据加工、监督微调、强化优化、离线评测”串成可复现的工程闭环。

项目强调三件事：

1. 数据合同优先：先保证数据格式和校验正确，再进行训练优化。
2. 评测驱动迭代：训练决策依赖固定评测集与统一指标，而不是随机试跑。
3. 脚本化复现：从 raw 数据到模型评估可通过 `scripts/` 快速重跑。

## 项目说明

项目支持以下典型场景：

1. 快速验证 LLM 是否能学会 2048 的基本动作策略。
2. 对比不同训练阶段（base / SFT / RL）的策略质量差异。
3. 通过固定 easy/medium/hard 评测集做可追踪的版本迭代。

当前默认基座模型示例为 `Qwen/Qwen3-1.7B`，训练方式包含：

1. `SFT`：`src/models/trl_train.py`
2. `ReST`：`src/models/modern_rl_trainer.py`
3. `GRPO`：`src/models/grpo.py`

## 项目结构

```text
.
├── src/
│   ├── data/                 # 数据生成、处理、契约、prompt 组装
│   │   ├── generator.py      # 生成 raw 游戏轨迹
│   │   ├── processor.py      # raw -> processed 并做数据校验
│   │   ├── contracts.py      # 数据 schema/合同定义
│   │   └── prompting.py      # 训练与推理 prompt 模板
│   ├── envs/
│   │   └── game_2048.py      # 2048 环境与动作规则
│   ├── models/
│   │   ├── trl_train.py      # SFT 训练入口
│   │   ├── modern_rl_trainer.py # ReST 训练
│   │   └── grpo.py           # GRPO 训练
│   ├── eval/
│   │   ├── evaluator.py      # 评测核心逻辑与指标聚合
│   │   └── eval_set_builder.py # 固定评测集构建
│   └── utils/                # 监控与通用工具
├── scripts/                  # 一键流程脚本（生成/训练/评测）
├── tests/                    # 单元测试与回归测试
├── docs/                     # 教学型流程文档与排障手册
└── data/                     # 数据与评测结果输出目录
```

## 端到端流程

```text
raw data generation
  -> processed dataset + validation
    -> SFT / RL training
      -> random eval + fixed baseline eval
        -> metrics artifacts (json/png)
```

这个项目用于复现一个完整的 2048 LLM 训练闭环：

1. 生成原始轨迹数据（raw）
2. 转换为训练数据（processed）
3. 训练 SFT / RL 模型
4. 用统一评测脚本对模型效果做可复现评估

如果你第一次上手，按本 README 的命令顺序执行即可复现。

## 快速复现（推荐先跑）

这条路径用于快速验证环境与主流程是否可用，几分钟能拿到模型和评测结果。

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) 跑最小端到端流程（数据 -> 训练 -> 评测）
bash scripts/smoke_pipeline.sh --monitor_backend none
```

执行完成后，重点看这几个目录：

1. `data/raw_smoke`：生成的原始轨迹
2. `data/processed_smoke`：训练集与验证集
3. `checkpoints/sft_smoke`：SFT 模型产物
4. `data/eval/smoke/eval_results.json`：评测结果

## 完整复现（数据生成到评测）

下面是完整流程，适合正式实验。

### 0) 环境准备

```bash
pip install -r requirements.txt
```

可选：先确认单元测试通过。

```bash
pytest -q tests
```

### 1) 生成 raw 数据

```bash
bash scripts/generate_data.sh \
  --num_games 2000 \
  --difficulty mixed \
  --output_dir data/raw \
  --seed 42
```

如果要生成更强监督数据，建议直接用 expert（expectimax）策略：

```bash
bash scripts/generate_data.sh \
  --num_games 2000 \
  --difficulty expert \
  --expert_depth 2 \
  --expert_max_empty 8 \
  --output_dir data/raw_expert \
  --seed 42
```

第一版（100% expert + CoT）可直接用：

```bash
bash scripts/generate_expert_cot_data.sh \
  --num_games 1000 \
  --expert_depth 3 \
  --expert_max_empty 6 \
  --output_dir data/raw_expert \
  --seed 42
```

### 2) raw -> processed 并校验数据合同

```bash
python -m src.data_gen.processor \
  --input_dir data/raw_expert \
  --output_dir data/processed \
  --use_thinking \
  --validate
```

### 3) 训练 SFT 模型

```bash
bash scripts/train.sh \
  --mode sft \
  --train_data data/processed/train \
  --val_data data/processed/val \
  --output_dir checkpoints/sft \
  --epochs 3 \
  --monitor_backend none
```

### 4) 构建固定评测集（easy/medium/hard）

```bash
bash scripts/build_eval_sets.sh \
  --raw_dir data/raw \
  --output_dir data/eval_sets \
  --seed 42
```

### 5) 评测模型（随机对局）

```bash
bash scripts/evaluate.sh \
  --model_path checkpoints/sft \
  --base_model Qwen/Qwen3-1.7B \
  --num_games 100 \
  --max_steps 1000 \
  --temperature 0.1 \
  --seed 42 \
  --output_dir data/eval/sft
```

### 6) 评测模型（固定基线集）

```bash
bash scripts/run_baseline_eval.sh \
  --model_path checkpoints/sft \
  --base_model Qwen/Qwen3-1.7B \
  --eval_set_dir data/eval_sets \
  --output_dir data/eval/baseline_sft \
  --max_steps 1000 \
  --temperature 0.1 \
  --seed 42 \
  --no_visualize
```

## 你会得到什么评测结果

### 单次评测产物

1. `data/eval/<run>/eval_results.json`
2. `data/eval/<run>/eval_results.png`

`eval_results.json` 关键字段：

1. `mean_score` / `median_score` / `std_score`
2. `mean_max_tile` / `max_tile_reached`
3. `legal_move_rate`
4. `mean_steps`
5. `scores` / `max_tiles` / `legal_move_ratios` / `steps`
6. `model_type` / `base_model` / `model_path` / `seed`

### 固定基线评测产物

1. `data/eval/baseline_<name>/easy/eval_results.json`
2. `data/eval/baseline_<name>/medium/eval_results.json`
3. `data/eval/baseline_<name>/hard/eval_results.json`
4. `data/eval/baseline_<name>/baseline_results.json`

`baseline_results.json` 包含各 split 指标和按样本数加权的 `overall` 指标，便于做版本比较。

### 评测集产物

1. `data/eval_sets/easy.jsonl`
2. `data/eval_sets/medium.jsonl`
3. `data/eval_sets/hard.jsonl`
4. `data/eval_sets/manifest.json`

`manifest.json` 记录样本规模、比例、seed、每个 bucket 的样本数，用于复现实验。

## 一键流程脚本

1. `scripts/smoke_pipeline.sh`：最小端到端验证
2. `scripts/generate_data.sh`：生成 raw 数据
3. `scripts/train.sh`：SFT/RL 训练入口
4. `scripts/evaluate.sh`：单模型评测
5. `scripts/build_eval_sets.sh`：构建固定评测集
6. `scripts/run_baseline_eval.sh`：easy/medium/hard 基线评测汇总
7. `scripts/train_sft_rest.sh`：SFT + ReST 流程
8. `scripts/train_sft_grpo.sh`：SFT + GRPO 流程

## 智能体评测（baseline + LLM thinking 开关）

新增统一评测入口：`python -m src.eval.agent_eval`，支持：

1. 规则智能体 baseline（`--agent_type rule`）
2. 纯随机智能体 baseline（`--agent_type random`）
3. LLM 智能体 thinking 开/关（`--agent_type llm --use_thinking/--no_thinking`）

每类都支持两种能力：

1. 快速评测（`--quick`）
2. 可视化单局回放（`--visualize_game`，终端逐步显示棋盘与动作）

示例命令：

```bash
# 1) 规则 baseline（快速）
python -m src.eval.agent_eval \
  --agent_type rule \
  --rule_difficulty advanced \
  --rule_with_thinking \
  --quick \
  --seed 42 \
  --output_dir data/eval_agents/rule_baseline

# 2) 随机 baseline（快速）
python -m src.eval.agent_eval \
  --agent_type random \
  --quick \
  --seed 42 \
  --output_dir data/eval_agents/random_baseline

# 3a) LLM（thinking 开）
python -m src.eval.agent_eval \
  --agent_type llm \
  --model_path checkpoints/sft \
  --use_thinking \
  --quick \
  --seed 42 \
  --output_dir data/eval_agents/llm_thinking_on

# 3b) LLM（thinking 关）
python -m src.eval.agent_eval \
  --agent_type llm \
  --model_path checkpoints/sft \
  --no_thinking \
  --quick \
  --seed 42 \
  --output_dir data/eval_agents/llm_thinking_off
```

## 推荐文档

1. 工作流总览：`docs/workflow/00_WORKFLOW_INDEX.md`
2. 入门教程：`docs/BEGINNER_WORKFLOW_TUTORIAL.md`
3. 评测基线设计：`docs/workflow/05_EVALUATION_BASELINES.md`
4. 开发与测试：`docs/DEVELOPMENT.md`
5. 模块说明：`src/data_gen/readme.md`、`src/envs/readme.md`、`src/models/readme.md`、`src/eval/readme.md`

## 注意事项

1. Mac 适合做数据与流程验证，长时训练建议在 NVIDIA GPU 服务器执行。
2. 做模型对比时请固定评测集、推理参数和 seed，否则结果不可比。


python -m src.data_gen.next_board_processor \
    --output_dir data/processed_next_board \
    --num_samples 4096 \
    --validate

## grpo

    CUDA_VISIBLE_DEVICES=0 python -m src.models.next_board_grpo \
    --model /home/cnz/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --input_dir data/processed_next_board/train \
    --output_dir checkpoints/grpo_next_board \
    --num_samples 4096 \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum 16 \
    --num_generations 2 \
    --lr 5e-6 \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --save_steps 0.25 \
    --logging_steps 5 \
    --monitor_backend tensorboard

  CUDA_VISIBLE_DEVICES=0 python -m src.models.next_board_grpo_lora \
    --model /home/cnz/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
    --input_dir data/processed_next_board/train \
    --output_dir checkpoints/grpo_next_board_lora \
    --num_samples 4096 \
    --epochs 1 \
    --batch_size 4 \
    --grad_accum 8 \
    --num_generations 2 \
    --lr 5e-6 \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --monitor_backend tensorboard

  参数含义简要如下：

  - --model：SFT 初始化模型或已有 checkpoint。
  - --input_dir：next-board 训练集目录，通常是 data/processed_next_board/train。
  - --output_dir：GRPO 输出目录。
  - --num_samples：从数据集中采样多少条 prompt 做训练。
  - --epochs：训练轮数。
  - --batch_size：每卡 batch size。
  - --grad_accum：梯度累积步数。
  - --num_generations：每个 prompt 采样多少个 completion 做 GRPO 对比。
  - --lr：学习率。
  - --max_prompt_length：prompt 最大长度。
  - --max_completion_length：生成 JSON 的最大长度。
  - --monitor_backend none：关闭 wandb/tensorboard。

  如果你想先做一个最小冒烟版，可以把它缩成：

  python -m src.models.next_board_grpo \
    --model checkpoints/sft_next_board \
    --input_dir data/processed_next_board/train \
    --output_dir /tmp/grpo_next_board_smoke \
    --num_samples 64 \
    --epochs 1 \
    --batch_size 2 \
    --grad_accum 2 \
    --num_generations 2 \
    --monitor_backend none



mkdir -p ~/flash_attn_wheel_factory

docker run --rm --gpus 0 \
  -v ~/flash_attn_wheel_factory:/output \
  nvidia/cuda:12.8.0-devel-ubuntu24.04 /bin/bash -c '
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      curl git build-essential python3.10 python3.10-venv python3.10-dev tzdata

    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/usr/local/bin" sh
    /usr/local/bin/uv venv --seed /tmp/build_env --python 3.10
    source /tmp/build_env/bin/activate

    uv pip install torch ninja packaging wheel

    export MAX_JOBS=4
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE

    pip wheel --no-build-isolation \
      git+https://github.com/Dao-AILab/flash-attention.git@v2.8.3
    cp /tmp/flash_attn*.whl /output/
  '


  <!-- # 1. 退出当前环境
conda deactivate

# 2. 移除整个环境 (假设环境名叫 rl)
conda remove --name rl --all -y

# 3. 重新创建一个干净的环境
conda create --name rl python=3.12 -y  # 建议用 3.10，对 vllm 和 torch 兼容性最稳

# 4. 激活并重新用 uv 安装
conda activate rl -->