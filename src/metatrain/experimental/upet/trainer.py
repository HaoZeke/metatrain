"""metatrain/experimental/upet/trainer.py

Simple trainer for UPET - focuses on torch.compile compatibility.
"""

from typing import Any, Dict, List, Literal, Union
from pathlib import Path
import logging
import math

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset

from metatensor.torch import join

from metatrain.utils.abc import ModelInterface, TrainerInterface
from metatrain.utils.data import Dataset
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from metatomic.torch import ModelOutput

from .model import UPETModel
from .documentation import TrainerHypers

logger = logging.getLogger(__name__)


def get_scheduler(optimizer, hypers, steps_per_epoch):
    """Cosine schedule with warmup."""
    total_steps = hypers["num_epochs"] * steps_per_epoch
    warmup_steps = int(hypers.get("warmup_fraction", 0.01) * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def collate_fn(batch, neighbor_list_options):
    """Collate batch with neighbor lists."""
    systems = []
    targets_dict = {}
    
    for sample in batch:
        system = sample["system"]
        system = get_system_with_neighbor_lists(system, neighbor_list_options)
        systems.append(system)
        
        field_names = sample._fields
        for field_name in field_names:
            if field_name == "system":
                continue
            value = sample[field_name]
            if field_name not in targets_dict:
                targets_dict[field_name] = []
            targets_dict[field_name].append(value)
    
    targets = {}
    for key, values in targets_dict.items():
        if len(values) > 0:
            targets[key] = join(values, axis="samples")
    
    return systems, targets


def compute_loss(predictions: Dict[str, Any], targets: Dict[str, Any]) -> torch.Tensor:
    """Simple MSE loss over all targets."""
    total_loss = None
    
    for target_name in targets.keys():
        if target_name in predictions:
            pred_block = predictions[target_name].block()
            target_block = targets[target_name].block()
            
            pred_values = pred_block.values
            target_values = target_block.values.to(pred_values.device, pred_values.dtype)
            
            mse = ((pred_values - target_values) ** 2).mean()
            
            if total_loss is None:
                total_loss = mse
            else:
                total_loss = total_loss + mse
    
    if total_loss is None:
        device = next(iter(predictions.values())).block().values.device
        return torch.tensor(0.0, device=device)
    
    return total_loss


class Trainer(TrainerInterface[TrainerHypers]):
    """Simple trainer for UPET models."""

    __checkpoint_version__ = 1

    def __init__(self, hypers: TrainerHypers) -> None:
        super().__init__(hypers)
        self._optimizer_state = None
        self._scheduler_state = None
        self._epoch = 0

    def train(
        self,
        model: ModelInterface,
        dtype: torch.dtype,
        devices: List[torch.device],
        train_datasets: List[Union[Dataset, Subset]],
        val_datasets: List[Union[Dataset, Subset]],
        checkpoint_dir: str,
    ) -> None:
        """Train the UPET model."""
        
        device = devices[0]
        
        # Move model to device/dtype FIRST
        model = model.to(device=device, dtype=dtype)
        
        # Now compile the inner UPET model if requested
        # This happens AFTER dtype conversion so dtypes are consistent
        should_compile = model.hypers.get("compile", False)
        if should_compile:
            logger.info("Compiling UPET model with torch.compile...")
            # Compile the inner model, not the wrapper
            model.model = torch.compile(model.model, mode="reduce-overhead")
            logger.info("Compilation complete (will trace on first forward pass)")
        
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0] if val_datasets else None
        
        neighbor_list_options = model.requested_neighbor_lists()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hypers.get("batch_size", 16),
            shuffle=True,
            collate_fn=lambda b: collate_fn(b, neighbor_list_options),
            num_workers=self.hypers.get("num_workers") or 0,
        )
        
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.hypers.get("batch_size", 16),
                shuffle=False,
                collate_fn=lambda b: collate_fn(b, neighbor_list_options),
                num_workers=self.hypers.get("num_workers") or 0,
            )
        
        optimizer = AdamW(
            model.parameters(),
            lr=self.hypers.get("learning_rate", 1e-4),
            weight_decay=self.hypers.get("weight_decay") or 0.0,
        )
        
        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)
        
        scheduler = get_scheduler(optimizer, self.hypers, len(train_loader))
        if self._scheduler_state is not None:
            scheduler.load_state_dict(self._scheduler_state)
        
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        num_epochs = self.hypers.get("num_epochs", 1000)
        log_interval = self.hypers.get("log_interval", 1)
        checkpoint_interval = self.hypers.get("checkpoint_interval", 100)
        grad_clip = self.hypers.get("grad_clip_norm", 1.0)
        
        start_epoch = self._epoch
        
        for epoch in range(start_epoch, num_epochs):
            self._epoch = epoch
            
            model.train()
            train_loss = 0.0
            
            for systems, targets in train_loader:
                optimizer.zero_grad()
                
                outputs = {
                    name: ModelOutput(per_atom=False)
                    for name in targets.keys()
                }
                
                predictions = model(systems, outputs, selected_atoms=None)
                loss = compute_loss(predictions, targets)
                
                loss.backward()
                
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            val_loss = None
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for systems, targets in val_loader:
                        outputs = {
                            name: ModelOutput(per_atom=False)
                            for name in targets.keys()
                        }
                        predictions = model(systems, outputs, selected_atoms=None)
                        loss = compute_loss(predictions, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % log_interval == 0:
                msg = f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.6f}"
                if val_loss is not None:
                    msg += f", val_loss={val_loss:.6f}"
                logger.info(msg)
                print(msg)
            
            if (epoch + 1) % checkpoint_interval == 0:
                self._optimizer_state = optimizer.state_dict()
                self._scheduler_state = scheduler.state_dict()
                self.save_checkpoint(model, checkpoint_path / f"epoch_{epoch + 1}.ckpt")
        
        # Load best model
        if best_state is not None:
            model.load_state_dict(best_state)
            logger.info(f"Using best model from epoch {best_epoch + 1}")
        
        # Save final checkpoint (without compilation for portability)
        self.save_checkpoint(model, checkpoint_path / "best.ckpt")

    def save_checkpoint(self, model: ModelInterface, path: Union[str, Path]) -> None:
        """Save checkpoint with model and trainer state."""
        checkpoint = model.get_checkpoint()
        
        checkpoint["trainer_state"] = {
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "epoch": self._epoch,
            "trainer_ckpt_version": self.__checkpoint_version__,
        }
        checkpoint["train_hypers"] = dict(self.hypers)
        checkpoint["architecture_name"] = "experimental.upet"
        
        torch.save(checkpoint, path)

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: Dict[str, Any],
        hypers: TrainerHypers,
        context: Literal["restart", "finetune"],
    ) -> "Trainer":
        trainer = cls(hypers)
        
        if context == "restart" and "trainer_state" in checkpoint:
            trainer_state = checkpoint["trainer_state"]
            trainer._optimizer_state = trainer_state.get("optimizer_state")
            trainer._scheduler_state = trainer_state.get("scheduler_state")
            trainer._epoch = trainer_state.get("epoch", 0)
        
        return trainer
