# envs 模块说明

`src/envs` 提供训练与评估共用的 2048 环境实现。  
本模块决定“状态定义、动作语义、奖励计算”的基础行为。

## 模块职责
- 实现 4x4 2048 棋盘环境
- 定义动作映射（ID 与中文方向）
- 提供模型输出文本到动作 ID 的解析函数

## 主要文件

### `src/envs/game_2048.py`
- 常量：
  - `ACTION_MAP`：`0=上, 1=右, 2=下, 3=左`
  - `ACTION_NAMES_ENG` / `ACTION_NAMES_CHI`：方向映射
- 核心类：`Game2048`
- 工具函数：`parse_action_from_text(text)`

## `Game2048` 关键接口

### 生命周期
- `Game2048(seed=None)`：初始化环境
- `reset()`：重置棋盘并返回初始状态字符串

### 交互
- `step(action)`：执行动作，返回
  - `next_state: str`
  - `reward: float`
  - `done: bool`
  - `score: int`
- `get_valid_actions()`：返回当前局面的合法动作 ID 列表

### 状态与统计
- `get_max_tile()`：当前最大数字
- `get_empty_cells()`：空格数量
- `clone()`：复制当前环境状态（用于模拟）

## 奖励与终止规则
- 合法移动：奖励为当前步分数增量
- 非法移动（棋盘不变化）：奖励 `-1`
- 终止条件：
  - 无空位，且
  - 横向/纵向都不存在可合并相邻格

## 输出状态格式
- `_get_state()` 返回 4x4 数组字符串，例如：
```text
[
[2, 0, 0, 2],
[0, 4, 0, 0],
[0, 0, 0, 0],
[0, 0, 0, 0]
]
```
- 该格式被 `data`/`models`/`eval` 模块直接复用。

## 文本动作解析
- `parse_action_from_non_think_text(text)`：只解析非 `<think>` 区域中的 JSON。
- `parse_action_from_text(text)`：严格复用上述 JSON 解析；解析失败返回 `-1`（非法动作）。
- 非 `<think>` 区 JSON 必须包含 `局面/判断/选择` 三个顶层键，且 `选择` 必须是合法方向并在 `判断` 中为 `true`。
- JSON 非法、schema 不匹配、或选择非法方向时，统一按非法动作处理。

## 常见问题与建议

### 1) 同 seed 结果不稳定
- 检查是否有其他模块改写全局随机状态。
- 当前实现使用全局随机种子，若多实例并发可能相互影响。

### 2) 动作解析偏向某方向
- 确保模型在非 `<think>` 区只输出合法 JSON 对象。
- 可在上层 prompt 里强化 JSON schema 与“禁止额外文本”约束。
