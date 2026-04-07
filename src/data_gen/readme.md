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
  - `build_assistant_content(action, state_text, use_thinking, thinking=None)`
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
  - `states[i]`：`state`, `action`, `action_id`, `action_json`, `score`, `step`, `thinking?`, `info?`

#### `states[i].info`（Human-Lite，可选）
- 设计目标：仅保留“人眼快速可识别”的局面信息，避免复杂搜索值。
- 字段结构：
  - `info_version`：当前固定 `human_lite_v1`
  - `board_observation`：棋盘直观信息
    - `max_tile`：当前最大数字
    - `max_tile_positions`：最大数字位置列表（0-based）
    - `empty_cells`：空位数量
    - `non_zero_cells`：非空位数量
    - `max_tile_in_corner`：最大数字是否在角落
    - `max_tile_corner_name`：角落名称（若不在角落则为 `null`）
  - `visible_patterns`：可见结构线索
    - `adjacent_equal_pairs_count`：相邻相等对数量（水平+垂直）
    - `near_merge_lines`：可见合并线索所在的行/列（如 `第1行`、`第3列`）
  - `action_space`：动作空间事实
    - `valid_actions`：合法动作ID列表
    - `valid_action_names`：合法动作中文名
    - `invalid_action_names`：非法动作中文名
  - `chosen`：最终动作与简短理由
    - `action_id` / `action_name`
    - `reason_tags`：`keep_corner_max` / `seek_merge` / `keep_space` / `avoid_dead_end`
    - `reason_text_short`：简短自然语言理由

#### 动作ID映射
- `0=上`, `1=右`, `2=下`, `3=左`

### 输出：processed 训练集
- 路径：`data/processed/{train,val,test}`
- 关键字段：
  - 必填：`text`
  - 可选元数据：`schema_version`, `source_game_id`, `source_step`
- 约束：
  - `text` 必须非空
  - 非 `<think>` 区域必须是合法 JSON 对象，且可被严格动作解析器解析
  - CoT 模式要求包含 `<think>...</think>`

## 典型调用链

1. `generator.py` 生成 raw  
2. `processor.py` 优先使用 raw 里的 `action_json` 构建 completion，并执行格式校验  
3. `models/*` 直接读取 `data/processed/train|val`

## 常见问题与排查

### 1) 训练输出不是合法 JSON
- 检查是否训练和推理都使用同一套模板（`prompting.py`）。
- 检查 `format_sample_text(..., apply_chat_template=True)` 是否与推理保持一致。

### 2) CoT 样本未生效
- 检查 processed 的 `text` 是否真的含 `<think>...</think>`，而不只是 raw 里有 `thinking` 字段。

### 3) 数据量看起来够但效果差
- 先跑 `processor --validate`，确认 `action/action_id` 对齐，`step` 连续，`score` 无回退。

## 扩展建议
- 新增字段时先改 `contracts.py`，再改 `processor.py` 和测试。
- 增加新模板时统一放在 `prompting.py`，避免训练/推理两套逻辑分叉。

我已经给你实现了一套基于 src/data_gen/next_board_processor.py 生成数据的短链 SFT 数据处理代码：

  - 新文件：src/data_gen/next_board_sft_processor.py
  - 测试：tests/test_next_board_sft_processor.py

  这套 SFT prompt 的核心设计是：

  Simulate exactly one 2048 move.

  The board is 4 rows by 4 columns: board[r][c].
  Row 0 is top, row 3 is bottom, column 0 is left, column 3 is right.

  Apply the rule to each affected line only:
  1. remove zeros
  2. merge adjacent equal tiles once
  3. pad zeros on the far side

  UP/DOWN operate on columns.
  LEFT/RIGHT operate on rows.
  Rebuild the final answer as 4 rows.

  If you think, keep it short:
  - line 1: which lines are affected
  - line 2: one or two example transformations
  - line 3: rebuild rows

  Output the final answer as strict JSON after </think>.

  assistant 目标会被构造成这种格式：

  <think>
  RIGHT uses rows.
  r0: [4, 8, 4, 0]->[0, 4, 8, 4]; r1: [4, 32, 32, 0]->[0, 0, 4, 64]
  Rebuild 4 rows -> [[0, 4, 8, 4], [0, 0, 4, 64], [0, 0, 0, 0], [0, 0, 0, 4]]
  </think>

  {"next_board": [[0, 4, 8, 4], [0, 0, 4, 64], [0, 0, 0, 0], [0, 0, 0, 4]]}

  我已经实际跑通了数据转换命令：

  python -m src.data_gen.next_board_sft_processor \
    --input_dir data/processed_next_board \
    --output_dir /tmp/processed_next_board_sft

  你可以正式生成到项目目录：

  python -m src.data_gen.next_board_sft_processor \
    --input_dir data/processed_next_board \
    --output_dir data/processed_next_board_sft

  然后直接用现有 SFT 训练入口训练：

  python -m src.models.trl_train \
    --mode sft \
    --model Qwen/Qwen3-1.7B-Instruct \
    --train_data data/processed_next_board_sft/train \
    --val_data data/processed_next_board_sft/val \
    --output_dir checkpoints/sft_next_board_short_cot \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 5e-5 \
    --monitor_backend none