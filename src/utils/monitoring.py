"""Monitoring backend helpers for training entrypoints."""

from __future__ import annotations

from typing import Dict, List, Optional


def normalize_monitor_backend(
    monitor_backend: Optional[str] = None,
    *,
    no_wandb: bool = False,
    use_wandb: Optional[bool] = None,
) -> str:
    """Resolve monitor backend with backward-compatible flags."""
    if monitor_backend:
        backend = monitor_backend.lower()
        if backend not in {"wandb", "tensorboard", "none"}:
            raise ValueError(f"Unsupported monitor backend: {monitor_backend}")
        return backend

    if use_wandb is not None:
        return "wandb" if use_wandb else "none"

    return "none" if no_wandb else "wandb"



def report_to_list(backend: str) -> List[str]:
    """Map backend to Transformers/TRL report_to list."""
    return [backend] if backend in {"wandb", "tensorboard"} else []


class IterationMonitor:
    """Simple scalar logger for outer-loop training stats (ReST style)."""

    def __init__(self, backend: str, log_dir: str):
        self.backend = backend
        self.log_dir = log_dir
        self._tb_writer = None

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=log_dir)

    def init(self, project: str, config: Dict):
        if self.backend == "wandb":
            import wandb

            wandb.init(project=project, config=config)

    def log(self, data: Dict, step: Optional[int] = None):
        if self.backend == "wandb":
            import wandb

            wandb.log(data, step=step)
        elif self.backend == "tensorboard" and self._tb_writer is not None:
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(key, value, global_step=step)

    def finish(self):
        if self.backend == "wandb":
            import wandb

            wandb.finish()
        elif self.backend == "tensorboard" and self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
