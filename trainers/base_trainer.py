"""
trainers/base_trainer.py  —  Template Method pattern
=====================================================

HOW IT WORKS:
  BaseTrainer owns the full training loop — AMP, scaler, scheduler,
  bad-batch skip, history lists, per-class tracking, checkpointing.

  HOOKS (methods subclasses override):
    move_input(x)        — tensor input vs dict input (GroupTrainer)
    compute_loss(out, y) — single vs multitask loss (PersonTrainer)
    compute_metrics(...) — accuracy vs F1 (PersonTrainer, GroupTrainer)
    print_epoch(...)     — 1-line vs 3-loss print (PersonTrainer)
    _checkpoint_name()   — filename for saved checkpoint

  PersonTrainer and GroupTrainer override ONLY their hooks.
  Everything else is inherited and runs identically for all baselines.
"""

import os
import torch
from torch import amp


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True


class BaseTrainer:
    """
    Template Method base for all trainers.

    Args:
        model           : nn.Module
        train_loader    : DataLoader
        val_loader      : DataLoader
        optimizer       : torch optimizer
        scheduler       : LR scheduler (CosineAnnealingLR)
        loss_fn         : callable
        accuracy        : metrics.accuracy(all_targets, all_preds) → float
        f1_score        : metrics.f1_calc(all_targets, all_preds) → float
        save_checkpoint : checkpoint.save_checkpoint(model, opt, epoch, val, path)
        device          : torch.device
        epochs          : int
        output_dir      : str — directory for saved checkpoints
        class_map       : dict {class_name: class_idx}
        print_perclass  : bool — print per-class breakdown each epoch 
    """

    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, loss_fn,
                 accuracy, f1_score, save_checkpoint,
                 device, epochs, output_dir, class_map,
                 print_perclass: bool = False):

        self.model           = model.to(device)
        self.train_loader    = train_loader
        self.val_loader      = val_loader
        self.optimizer       = optimizer
        self.scheduler       = scheduler
        self.loss_fn         = loss_fn
        self.accuracy        = accuracy
        self.f1_score        = f1_score
        self.save_checkpoint = save_checkpoint
        self.device          = device
        self.epochs          = epochs
        self.output_dir      = output_dir
        self.class_map       = class_map
        self.print_perclass  = print_perclass

        self.use_amp = (device.type == "cuda")
        self.scaler  = amp.GradScaler(enabled=self.use_amp)

        self.best_metric  = 0.0
        self.train_losses = []
        self.val_metrics  = []
        self.lr_history   = []

    # =========================================================================
    #  TEMPLATE — the main loop.  Never override this.
    # =========================================================================
    def train(self):
        for epoch in range(self.epochs):
            train_result = self._run_epoch(self.train_loader, training=True)
            val_result   = self._run_epoch(self.val_loader,   training=False)

            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.train_losses.append(train_result["loss"])
            self.val_metrics.append(val_result["metric"])
            self.lr_history.append(lr)

            self.print_epoch(epoch, lr, train_result, val_result)   # HOOK

            if val_result["metric"] >= self.best_metric:
                self.best_metric = val_result["metric"]
                path = os.path.join(self.output_dir, self._checkpoint_name())
                self.save_checkpoint(
                    self.model, self.optimizer,
                    epoch, self.best_metric, path
                )
                print(f"==> New Best: {self.best_metric:.4f}  saved → {path}")

    # =========================================================================
    #  TEMPLATE — one epoch (train or val).  Never override this.
    # =========================================================================
    def _run_epoch(self, loader, training: bool) -> dict:
        self.model.train(training)

        total_loss    = 0.0
        batch_count   = 0
        all_preds     = []
        all_targets   = []
        class_count   = torch.zeros(len(self.class_map), device=self.device)
        class_correct = torch.zeros(len(self.class_map), device=self.device)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x, y in loader:
                # Skip bad batches — size-1 batch crashes BatchNorm
                if y is None or y.size(0) <= 1:
                    continue

                x = self.move_input(x)                          # HOOK 1
                y = y.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(x)
                    loss, preds, targets = self.compute_loss(outputs, y)  # HOOK 2

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                all_preds.append(preds.detach())
                all_targets.append(targets.detach())
                total_loss  += loss.item()
                batch_count += 1

                if self.print_perclass:
                    for name, cls in self.class_map.items():
                        mask = (targets == cls)
                        class_count[cls]   += mask.sum()
                        class_correct[cls] += (preds[mask] == cls).sum()

        if self.print_perclass:
            phase = "TRAIN" if training else "VAL"
            self._print_perclass(class_count, class_correct, phase)

        metric = self.compute_metrics(all_preds, all_targets)   # HOOK 3

        return {
            "loss":    total_loss / max(batch_count, 1),
            "metric":  metric,
            "preds":   all_preds,
            "targets": all_targets,
        }

    # =========================================================================
    #  HOOKS — defaults here.  Override in subclasses only what differs.
    # =========================================================================

    def move_input(self, x):
        """
        HOOK 1: move x to device.
        Default  → x is a plain tensor.
        GroupTrainer overrides → x is a dict of tensors.
        """
        return x.to(self.device, non_blocking=True)

    def compute_loss(self, outputs, y):
        """
        HOOK 2: compute loss, extract predictions and targets.
        Default  → single classification loss, preds from argmax.
        PersonTrainer overrides → multitask, unpacks (motion, action) outputs.

        Must return: (scalar_loss, preds_LongTensor[B], targets_LongTensor[B])
        """
        loss  = self.loss_fn(outputs, y)
        preds = outputs.argmax(dim=1)
        return loss, preds, y

    def compute_metrics(self, all_preds: list, all_targets: list) -> float:
        """
        HOOK 3: compute the metric used for checkpointing.
        Default  → accuracy.
        PersonTrainer & GroupTrainer override → macro F1.
        """
        return self.accuracy(all_targets, all_preds)

    def print_epoch(self, epoch: int, lr: float,
                    train: dict, val: dict):
        """
        HOOK 4: what to print at the end of each epoch.
        Default  → one loss line + one metric line.
        PersonTrainer overrides → three loss values (total, action, motion).
        """
        print(f"\nEpoch [{epoch}] | lr: {lr:.7f}")
        print(f"TRAIN → loss: {train['loss']:.3f} | metric: {train['metric']:.3f}")
        print(f"VAL   → loss: {val['loss']:.3f}   | metric: {val['metric']:.3f}")

    def _checkpoint_name(self) -> str:
        """
        HOOK 5: filename for saved checkpoint.
        Default  → best_model.pth
        PersonTrainer → PersonModel_best.pth
        GroupTrainer  → best_group_model.pth
        """
        return "best_model.pth"

    # =========================================================================
    #  PRIVATE UTILITY 
    # =========================================================================
    def _print_perclass(self, count: torch.Tensor,
                        correct: torch.Tensor, phase: str):
        print(f"\n--- {phase} Per-Class ---")
        for name, i in self.class_map.items():
            acc = 100.0 * correct[i] / (count[i] + 1e-6)
            print(f"  {name:12s}: n={int(count[i]):5d} | acc={acc:.1f}%")