"""Custom callbacks for training monitoring."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import lightning as L
import math
import torch


class MetricsSaveCallback(L.Callback):
    """Save training metrics to a single JSON file, updated in real-time."""

    def __init__(
        self, 
        save_dir: str = "./metrics", 
        save_every_n_steps: int = 10,
        log_grad_norm: bool = True,
    ):
        """
        Args:
            save_dir: Directory to save metric files
            save_every_n_steps: Save to disk every N steps (default: 10)
            log_grad_norm: Whether to compute and log gradient norm
        """
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_n_steps = save_every_n_steps
        self.log_grad_norm = log_grad_norm
        self.metrics_history = []
        self.metric_file: Optional[Path] = None
        self.start_timestamp: Optional[str] = None

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Create the metrics file at the start of training."""
        self.start_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metric_file = self.save_dir / f"training_metrics_{self.start_timestamp}.json"
        print(f"Metrics will be saved to: {self.metric_file}")
    
    def _compute_grad_norm(self, pl_module: L.LightningModule) -> float:
        """Compute total gradient norm across all parameters."""
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Collect metrics after each training batch and save periodically."""
        metrics = trainer.callback_metrics
        
        if metrics:
            metric_data = {
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                'batch_idx': batch_idx,
                'metrics': {k: v.item() if hasattr(v, 'item') else float(v) 
                           for k, v in metrics.items()}
            }
            
            # Compute and add gradient norm
            if self.log_grad_norm:
                grad_norm = self._compute_grad_norm(pl_module)
                metric_data['metrics']['grad_norm'] = grad_norm
            
            self.metrics_history.append(metric_data)
            
            # Save to disk every N steps
            if trainer.global_step % self.save_every_n_steps == 0:
                self._save_metrics(trainer)

    def _save_metrics(self, trainer: L.Trainer) -> None:
        """Write current metrics history to file."""
        if self.metric_file and self.metrics_history:
            output_data = {
                'timestamp': self.start_timestamp,
                'current_epoch': trainer.current_epoch,
                'current_step': trainer.global_step,
                'metrics_history': self.metrics_history
            }
            with open(self.metric_file, 'w') as f:
                json.dump(output_data, f, indent=2)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Final save at the end of training."""
        self._save_metrics(trainer)
        print(f"Training metrics saved to: {self.metric_file}")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log validation summary at end of each epoch."""
        metrics = trainer.callback_metrics
        if metrics:
            print(f"Epoch {trainer.current_epoch}: {metrics}")


class WarmupCosineLRScheduler(L.Callback):
    """Custom LR scheduler with linear warmup followed by cosine annealing."""

    def __init__(
        self,
        warmup_steps: int = 1000,
        base_lr: float = 5e-5,
        min_lr: float = 1e-6,
        total_steps: Optional[int] = None,
    ):
        """
        Args:
            warmup_steps: Number of steps for linear warmup
            base_lr: Base learning rate after warmup
            min_lr: Minimum learning rate at the end
            total_steps: Total training steps (if None, will be auto-calculated)
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Set total steps if not provided."""
        if self.total_steps is None:
            # Note: trainer.max_steps is -1 when not set, so we need to check for > 0
            if trainer.max_steps is not None and trainer.max_steps > 0:
                self.total_steps = trainer.max_steps
            else:
                self.total_steps = trainer.max_epochs * len(trainer.train_dataloader)
        print(f"LR Scheduler: warmup_steps={self.warmup_steps}, total_steps={self.total_steps}")

    def on_train_batch_start(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update learning rate at the start of each batch."""
        current_step = trainer.global_step

        if current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (current_step + 1) / self.warmup_steps
        else:
            # Cosine annealing
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            cos_value = math.cos(math.pi * progress)
            lr = self.min_lr + (self.base_lr - self.min_lr) * (1 + cos_value) / 2

        # Set LR for all parameter groups
        for param_group in trainer.optimizers[0].param_groups:
            param_group['lr'] = lr

        # Log current LR
        pl_module.log("lr", lr, prog_bar=True, on_step=True)
