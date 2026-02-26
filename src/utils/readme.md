# utils 模块说明

`src/utils` 提供跨模块复用的通用能力，当前以监控与实验记录适配为主。

## 模块职责
- 统一监控后端配置（wandb/tensorboard/none）
- 提供训练迭代日志写入封装
- 降低 `models` 模块中与监控平台耦合的代码量

## 主要文件

### `src/utils/monitoring.py`
- `normalize_monitor_backend(...)`
  - 将旧参数（如 `use_wandb`, `no_wandb`）归一化为统一 backend 字段。
- `report_to_list(backend)`
  - 适配 TRL/Transformers 的 `report_to` 参数格式。
- `IterationMonitor`
  - 统一初始化、逐步记录、结束清理接口。
  - 便于 ReST 等迭代流程持续记录指标。

## 设计原则
- 上层业务只关心“记录什么”，不关心“后端怎么接”。
- 对不存在的后端做可控降级（例如设为 `none`）。

## 使用建议
- 新训练器接入时，优先复用 `normalize_monitor_backend + report_to_list`。
- 自定义指标统一走 `IterationMonitor.log`，避免散落在业务代码。
