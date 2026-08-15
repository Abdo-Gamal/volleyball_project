"""
trainers/group_trainer.py
=========================
Replaces: b3_malti_input_trainer.py  (typo fixed: malti → multi)

Overrides three hooks from BaseTrainer:
  move_input      → x is a dict {"persons": tensor, "positions": tensor}
  compute_metrics → checkpoints by macro F1
  print_epoch     → shows acc and F1
  _checkpoint_name → 'best_group_model.pth'

Everything else is inherited from BaseTrainer unchanged.
"""

import torch
from trainers.base_trainer import BaseTrainer


class GroupTrainer(BaseTrainer):
    """
    Trainer for Baseline 3 GroupModel.
    Input x is a dict — override move_input to handle that.

    Usage in notebook:
        from trainers.group_trainer import GroupTrainer
        trainer = GroupTrainer(
            model=model, train_loader=trainloader, val_loader=valloader,
            optimizer=optimizer, scheduler=scheduler,
            loss_fn=FocalLoss(gamma=gamma),
            accuracy=accuracy, f1_score=f1_calc,
            save_checkpoint=save_checkpoint,
            device=device, epochs=epochs, output_dir=output_dir,
            class_map=GROUP_ACTION_TO_IDX, print_perclass=True,
        )
        trainer.train()
    """

    # ── HOOK 1: x is a dict, not a tensor ────────────────────────────────────
    def move_input(self, x):
        """
        GroupModel receives {"persons": Tensor[B,N,3,H,W],
                             "positions": Tensor[B,N,2]}.
        Move every value to device.
        """
        return {k: v.to(self.device, non_blocking=True) for k, v in x.items()}

    # ── HOOK 3: checkpoint by F1 ──────────────────────────────────────────────
    def compute_metrics(self, all_preds: list, all_targets: list) -> float:
        return self.f1_score(all_targets, all_preds)

    # ── HOOK 4: print acc and F1 ──────────────────────────────────────────────
    def print_epoch(self, epoch: int, lr: float,
                    train: dict, val: dict):
        train_acc = self.accuracy(train["targets"], train["preds"])
        val_acc   = self.accuracy(val["targets"],   val["preds"])

        print(f"\nEpoch [{epoch}] | lr: {lr:.7f}")
        print(
            f"TRAIN → loss: {train['loss']:.3f} | "
            f"acc: {train_acc:.3f} | F1: {train['metric']:.3f}"
        )
        print(
            f"VAL   → loss: {val['loss']:.3f}   | "
            f"acc: {val_acc:.3f}   | F1: {val['metric']:.3f}"
        )

    # ── HOOK 5: checkpoint filename ───────────────────────────────────────────
    def _checkpoint_name(self) -> str:
        return "best_group_model.pth"