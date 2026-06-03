"""Custom callbacks for training monitoring."""

import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import lightning as L


class RunManager(L.Callback):
    """Manage run directory: create outputs/{run_name}_{timestamp}/, save config and checkpoints."""

    def __init__(self, run_name: str = "run", base_dir: str = "./outputs", config_path: Optional[str] = None):
        super().__init__()
        self.run_name = run_name
        self.base_dir = Path(base_dir)
        self.config_path = Path(config_path) if config_path else None
        self.run_dir: Optional[Path] = None

    @staticmethod
    def _detect_config_from_argv() -> Optional[Path]:
        """Find the --config <file> argument from sys.argv (LightningCLI convention)."""
        import sys
        args = sys.argv
        for i, arg in enumerate(args):
            if arg in ("--config", "-c") and i + 1 < len(args):
                return Path(args[i + 1])
            if arg.startswith("--config="):
                return Path(arg.split("=", 1)[1])
        # Fallback: any positional .yaml argument
        for arg in args:
            if arg.endswith(".yaml") and Path(arg).exists():
                return Path(arg)
        return None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if stage != "fit" or self.run_dir is not None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_dir / f"{self.run_name}_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Copy config: prefer explicit path, fall back to auto-detection from argv
        config_to_copy = self.config_path
        if config_to_copy is None or not config_to_copy.exists():
            config_to_copy = self._detect_config_from_argv()
        if config_to_copy and config_to_copy.exists():
            shutil.copy(config_to_copy, self.run_dir / "config.yaml")
            print(f"Config saved: {self.run_dir / 'config.yaml'}")

        # Configure ModelCheckpoint to save in run_dir
        for cb in trainer.callbacks:
            if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint):
                cb.dirpath = str(self.run_dir)

        print(f"Run directory: {self.run_dir}")


class MetricsSaveCallback(L.Callback):
    """Save training metrics to JSON in outputs folder."""

    def __init__(self, save_every_n_steps: int = 10, log_grad_norm: bool = True):
        super().__init__()
        self.save_every_n_steps = save_every_n_steps
        self.log_grad_norm = log_grad_norm
        self.metrics_history = []
        self.metric_file: Optional[Path] = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Save metrics.json to base_dir (outside run_dir)
        base_dir = Path("./outputs")
        run_name = "run"
        for cb in trainer.callbacks:
            if isinstance(cb, RunManager):
                base_dir = cb.base_dir
                run_name = cb.run_dir.name if cb.run_dir else cb.run_name
                break
        base_dir.mkdir(parents=True, exist_ok=True)
        self.metric_file = base_dir / f"metrics_{run_name}.json"

    def on_train_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, 
                           outputs: Any, batch: Any, batch_idx: int) -> None:
        metrics = trainer.callback_metrics
        if not metrics:
            return
        
        data = {
            "epoch": trainer.current_epoch,
            "step": trainer.global_step,
            "metrics": {k: v.item() if hasattr(v, "item") else float(v) for k, v in metrics.items()}
        }
        if self.log_grad_norm:
            data["metrics"]["grad_norm"] = sum(
                p.grad.norm(2).item() ** 2 for p in pl_module.parameters() if p.grad is not None
            ) ** 0.5
        
        self.metrics_history.append(data)
        if trainer.global_step % self.save_every_n_steps == 0:
            self._save()

    def _save(self) -> None:
        if self.metric_file and self.metrics_history:
            with open(self.metric_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._save()


class WarmupCosineLR(L.Callback):
    """Linear warmup + cosine annealing LR scheduler."""

    def __init__(self, warmup_steps: int = 1000, base_lr: Optional[float] = None,
                 min_lr: Optional[float] = None, total_steps: Optional[int] = None):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        # Fall back to optimizer initial lr when base_lr is not set explicitly —
        # this lets HPO pass lr via --optimizer.init_args.lr without touching callbacks.
        if self.base_lr is None:
            self.base_lr = trainer.optimizers[0].param_groups[0]["lr"]
        if self.min_lr is None:
            self.min_lr = self.base_lr * 0.1
        if self.total_steps is None:
            self.total_steps = (trainer.max_steps if trainer.max_steps > 0
                               else trainer.max_epochs * len(trainer.train_dataloader))

    def on_train_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, 
                             batch: Any, batch_idx: int) -> None:
        step = trainer.global_step
        if step < self.warmup_steps:
            lr = self.base_lr * (step + 1) / self.warmup_steps
        else:
            progress = min((step - self.warmup_steps) / (self.total_steps - self.warmup_steps), 1.0)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
        
        for pg in trainer.optimizers[0].param_groups:
            pg["lr"] = lr
        pl_module.log("lr", lr, prog_bar=True, on_step=True)
