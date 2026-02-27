# Scripts Workflow Guide

这个文档只讲一件事: 如何主要通过 `scripts/*.sh` 跑通“数据生成 -> 训练 -> 评测”的全流程。

## 1) 前置准备

在项目根目录执行:

```bash
pip install -r requirements.txt
```

建议先做环境体检:

```bash
bash scripts/doctor.sh
```

## 2) 先跑最小闭环（推荐）

如果你第一次跑，先用 smoke 流程验证环境和主链路:

```bash
bash scripts/smoke_pipeline.sh --monitor_backend none --num_games 20 --epochs 1 --eval_games 5
```

这条命令会自动完成:
1. 生成少量 raw 数据
2. raw -> processed
3. SFT 训练
4. 评测并输出结果

输出目录:
1. `data/raw_smoke`
2. `data/processed_smoke`
3. `checkpoints/sft_smoke`
4. `data/eval/smoke`

## 3) 标准 SFT 全流程（模块化）

### Step A: 生成 raw 数据

```bash
bash scripts/generate_data.sh \
  --num_games 2000 \
  --difficulty mixed \
  --output_dir data/raw \
  --seed 42
```

难度可选值:
`random | basic | intermediate | advanced | expert | mixed`

如果要 100% expert + CoT 数据:

```bash
bash scripts/generate_expert_cot_data.sh \
  --num_games 2000 \
  --expert_depth 2 \
  --expert_max_empty 8 \
  --output_dir data/raw_expert \
  --seed 42
```

### Step B: raw -> processed（训练前必须）

当前项目没有独立 `scripts/process_data.sh`，这里使用项目标准命令:

```bash
python -m src.data_gen.processor \
  --input_dir data/raw \
  --output_dir data/processed \
  --use_thinking \
  --validate
```

### Step C: 训练 SFT

```bash
bash scripts/train.sh \
  --mode sft \
  --train_data data/processed/train \
  --val_data data/processed/val \
  --output_dir checkpoints/sft \
  --epochs 3 \
  --monitor_backend none
```

### Step D: 随机评测

```bash
bash scripts/evaluate.sh \
  --model_path checkpoints/sft \
  --base_model Qwen/Qwen3-1.7B \
  --num_games 100 \
  --temperature 0.1 \
  --seed 42 \
  --output_dir data/eval/sft
```

### Step E: 固定基线评测（推荐用于版本对比）

先构建评测集:

```bash
bash scripts/build_eval_sets.sh \
  --raw_dir data/raw \
  --output_dir data/eval_sets \
  --seed 42
```

再跑 easy/medium/hard 汇总评测:

```bash
bash scripts/run_baseline_eval.sh \
  --model_path checkpoints/sft \
  --eval_set_dir data/eval_sets \
  --output_dir data/eval/baseline_sft \
  --seed 42 \
  --no_visualize
```

## 4) 一键训练增强流程（全 bash 入口）

如果你希望尽量用 bash 脚本自动串起来，而不是手动分步骤，直接用下面两个入口:

### SFT + ReST

```bash
bash scripts/train_sft_rest.sh --monitor_backend none
```

### SFT + GRPO

```bash
bash scripts/train_sft_grpo.sh --monitor_backend none
```

这两个脚本会自动处理数据准备、SFT、RL 阶段和评测。

## 5) 日常开发常用脚本

```bash
bash scripts/lint.sh
bash scripts/typecheck.sh
bash scripts/test.sh
bash scripts/ci_local.sh --skip-smoke
```

## 6) 结果怎么看

重点看这些文件:
1. `data/eval/*/eval_results.json`
2. `data/eval/*/eval_results.png`
3. `data/eval/baseline_*/baseline_results.json`

## 7) 常见问题

1. `train.sh` 报找不到 `data/processed/train`
   先执行 `python -m src.data_gen.processor ...`，因为训练脚本读取的是 processed 数据。

2. 显存不足
   降低 `--batch_size`，或在训练脚本中关闭/调整量化相关选项。

3. 想查看每个脚本参数
   已在每个 `scripts/*.sh` 文件头部添加“参数说明区”，直接打开脚本即可查到默认值和可选值。
