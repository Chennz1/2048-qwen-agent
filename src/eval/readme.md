# eval 模块说明

`src/eval` 负责离线评估模型策略质量，给出可比较的指标结果。  
评估逻辑与训练逻辑复用同一环境与 prompt 体系。

## 模块职责
- 执行多局对战评估
- 聚合性能指标并输出结果文件
- 支持标准推理与 vLLM 加速推理

## 文件结构与作用

### `src/eval/evaluator.py`
- 核心类：`Game2048Evaluator`
- 关键方法：
  - `predict_action(...)`：单步动作预测（标准/vLLM 两种实现）
  - `evaluate(...)`：多局评估并返回统计指标
- 典型指标：
  - 平均分、分数分布
  - 最大方块统计
  - 合法动作率
  - 平均步数

### `src/eval/eval_set_builder.py`
- 用于构建固定评估集，支持写入 `manifest.json`，便于版本追踪与复现实验。

## 典型用法

```bash
python -m src.eval.evaluator \
  --model_path ./checkpoints/sft \
  --base_model Qwen/Qwen3-1.7B
```

可选：
- `--is_base_model`：仅评估基座模型
- `--use_vllm`：启用 vLLM
- `--load_in_4bit / --load_in_8bit`：标准模式量化加载

## 推理模式说明

### 标准模式（Transformers）
- 兼容性最好，适合开发调试与小规模评估。
- 支持 LoRA 权重加载（`PeftModel`）。

### vLLM 模式
- 适合大规模评估，吞吐更高。
- 依赖环境准备更严格（CUDA/vLLM 版本匹配）。

## 输出结果建议
- 保留原始评估 JSON（含参数配置）
- 记录评估时间、模型版本、数据版本（推荐含 manifest 引用）
- 对比实验时固定随机种子与局数

## 常见问题

### 1) 输出很长文本而不是动作
- 检查 prompt 是否强约束“最终只输出一个动作字”。
- 检查动作解析是否被非动作文本干扰。

### 2) 基座和微调模型差异不明显
- 增加评估局数，降低方差。
- 固定评估集，避免每轮随机局面变化掩盖差异。
