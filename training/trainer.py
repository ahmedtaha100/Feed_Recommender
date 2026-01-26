import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_checkpoint_safe(path, device=None):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        import numpy._core.multiarray
        torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except Exception:
            logger.warning("Loading checkpoint with weights_only=False (legacy checkpoint)")
            return torch.load(path, map_location=device, weights_only=False)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = "checkpoints",
        log_interval: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.mixed_precision = mixed_precision and self.device.type == "cuda"
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)

        self.scaler = GradScaler() if self.mixed_precision else None
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.training_history: List[Dict] = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            batch = self._move_batch_to_device(batch)

            if self.mixed_precision:
                with autocast():
                    outputs = self._forward_batch(batch)
                    loss = outputs["loss"] / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()
            else:
                outputs = self._forward_batch(batch)
                loss = outputs["loss"] / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.scheduler is not None:
                        self.scheduler.step()

            total_loss += outputs["loss"].item()
            num_batches += 1
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{lr:.2e}"})

        return {"train_loss": total_loss / num_batches}

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader,
    ) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = self._move_batch_to_device(batch)
            outputs = self._forward_batch(batch)
            total_loss += outputs["loss"].item()
            num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 3,
    ) -> Dict[str, List[float]]:
        logger.info(f"Training on device: {self.device}")
        logger.info(f"Mixed precision: {self.mixed_precision}")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            eval_metrics = self.evaluate(eval_loader)

            metrics = {**train_metrics, **eval_metrics, "epoch": epoch}
            self.training_history.append(metrics)

            logger.info(
                f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f}, "
                f"val_loss={metrics['val_loss']:.4f}"
            )

            if eval_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = eval_metrics["val_loss"]
                self.save_checkpoint("best_model.pt")
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(f"epoch_{epoch}.pt")

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        return self.training_history

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(
            history_title=batch["history_title"],
            history_category=batch["history_category"],
            history_mask=batch["history_mask"],
            pos_title=batch["pos_title"],
            pos_abstract=batch["pos_abstract"],
            pos_category=batch["pos_category"],
            pos_subcategory=batch["pos_subcategory"],
            neg_title=batch.get("neg_title"),
            neg_abstract=batch.get("neg_abstract"),
            neg_category=batch.get("neg_category"),
            neg_subcategory=batch.get("neg_subcategory"),
        )

    def _move_batch_to_device(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def save_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": int(self.global_step),
            "best_val_loss": float(self.best_val_loss),
            "training_history": self.training_history,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, filename: str) -> None:
        path = self.checkpoint_dir / filename
        checkpoint = load_checkpoint_safe(path, self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint.get("training_history", [])
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info(f"Checkpoint loaded from {path}")


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(1, self.warmup_steps)
        else:
            progress = (self.last_epoch - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            scale = max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return [
            self.min_lr + (base_lr - self.min_lr) * scale
            for base_lr in self.base_lrs
        ]


@torch.no_grad()
def extract_item_embeddings(
    model: nn.Module,
    news_data: Dict,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, Dict[int, int]]:
    model.eval()
    model.to(device)

    news_items = [(nid, data) for nid, data in news_data.items() if data["news_idx"] > 0]

    embeddings = []
    idx_mapping = {}

    for i in tqdm(range(0, len(news_items), batch_size), desc="Extracting embeddings"):
        batch_items = news_items[i:i+batch_size]

        titles = torch.tensor([item[1]["title"] for item in batch_items], dtype=torch.long)
        abstracts = torch.tensor([item[1]["abstract"] for item in batch_items], dtype=torch.long)
        categories = torch.tensor([item[1]["category"] for item in batch_items], dtype=torch.long)
        subcategories = torch.tensor([item[1]["subcategory"] for item in batch_items], dtype=torch.long)

        titles = titles.to(device)
        abstracts = abstracts.to(device)
        categories = categories.to(device)
        subcategories = subcategories.to(device)

        item_emb = model.encode_item(
            title=titles,
            abstract=abstracts,
            category=categories,
            subcategory=subcategories,
        )

        embeddings.append(item_emb.cpu().numpy())

        for j, (nid, data) in enumerate(batch_items):
            idx_mapping[data["news_idx"]] = len(idx_mapping)

    embeddings = np.vstack(embeddings)
    return embeddings, idx_mapping
