# data 模块说明

`src/data` 负责整个训练链路最关键的一层：数据契约统一、样本生成、格式转换与校验。  
如果训练效果异常，优先排查本模块。

## 模块职责
- 定义并维护 raw/processed 数据契约
- 生成 2048 轨迹数据（可带 CoT 思维文本）
- 将 raw 数据转换为训练用 `text` 样本
- 在转换阶段做 schema 校验和质量分析

## 文件结构与作用

### `src/data_gen/contracts.py`
- 作用：数据 schema 的单一真源（Single Source of Truth）。
- 关键常量：
  - `SCHEMA_VERSION`
  - `VALID_ACTIONS`（`上/右/下/左`）
  - `VALID_ACTION_IDS`（`0/1/2/3`）
- 关键函数：
  - `validate_raw_game(game)`：校验单局 raw 数据
  - `validate_processed_sample(sample, require_thinking=False)`：校验训练样本
  - `summarize_validation_failures(errors)`：汇总错误信息

### `src/data_gen/prompting.py`
- 作用：统一训练与推理的 prompt 构造方式，避免模板漂移。
- 关键函数：
  - `build_user_prompt(state_text, use_thinking)`
  - `build_assistant_content(action, use_thinking, thinking=None)`
  - `build_messages(...)`
  - `format_sample_text(tokenizer, messages, apply_chat_template)`
  - `format_inference_prompt(tokenizer, state_text, use_thinking=True)`

### `src/data_gen/generator.py`
- 作用：从环境策略生成 raw 轨迹数据文件。
- 典型产物：`data/raw/game_000001.json`
- 常见入口：
  - `generate_games(...)`
  - `generate_mixed_data(...)`
  - `save_games(games, output_dir)`

### `src/data_gen/processor.py`
- 作用：`raw -> processed`，并切分 `train/val/test`。
- 关键流程：
  - 读取 raw JSON
  - 校验 raw 数据（可严格模式）
  - 构造 messages
  - 生成 `text` 字段（可应用 chat_template）
  - 再次校验 processed 样本
  - 保存 Dataset 与 manifest
- 常见入口：
  - `games_to_training_format(...)`
  - `split_dataset(...)`
  - `save_datasets(...)`
  - `save_processing_manifest(...)`

## 数据输入输出契约

### 输入：raw 游戏文件
- 路径：`data/raw/*.json`
- 关键字段：
  - 顶层：`schema_version`, `game_id`, `difficulty`, `states`, `final_score`, `max_tile`, `total_steps`
  - `states[i]`：`state`, `action`, `action_id`, `score`, `step`, `thinking?`

### 输出：processed 训练集
- 路径：`data/processed/{train,val,test}`
- 关键字段：
  - 必填：`text`
  - 可选元数据：`schema_version`, `source_game_id`, `source_step`
- 约束：
  - `text` 必须非空
  - 结尾必须是合法动作之一：`上/右/下/左`
  - CoT 模式要求包含 `<think>...</think>`

## 典型调用链

1. `generator.py` 生成 raw  
2. `processor.py` 执行格式转换与校验  
3. `models/*` 直接读取 `data/processed/train|val`

## 常见问题与排查

### 1) 训练输出不是动作字符
- 检查是否训练和推理都使用同一套模板（`prompting.py`）。
- 检查 `format_sample_text(..., apply_chat_template=True)` 是否与推理保持一致。

### 2) CoT 样本未生效
- 检查 processed 的 `text` 是否真的含 `<think>...</think>`，而不只是 raw 里有 `thinking` 字段。

### 3) 数据量看起来够但效果差
- 先跑 `processor --validate`，确认 `action/action_id` 对齐，`step` 连续，`score` 无回退。

## 扩展建议
- 新增字段时先改 `contracts.py`，再改 `processor.py` 和测试。
- 增加新模板时统一放在 `prompting.py`，避免训练/推理两套逻辑分叉。
