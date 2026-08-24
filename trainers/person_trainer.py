"""
trainers/person_trainer.py
==========================

Overrides three hooks from BaseTrainer:
  compute_loss   → unpacks (motion_logits, action_logits), returns 4-tuple
  compute_metrics → checkpoints by macro F1, not accuracy
  print_epoch    → prints total + action + motion losses separately
  _checkpoint_name → 'PersonModel_best.pth'

Everything else (AMP, scaler, scheduler, bad-batch skip,
per-class tracking, history lists) is inherited from BaseTrainer.
"""

import torch
from trainers.base_trainer import BaseTrainer

class PersonTrainer(BaseTrainer):        

    """
    Trainer for Baseline 3 PersonModel (multitask: motion + action).

    Usage in notebook:
        from trainers.person_trainer import PersonTrainer
        trainer = PersonTrainer(
            model=model, train_loader=trainloader, val_loader=valloader,
            optimizer=optimizer, scheduler=scheduler,
            loss_fn=MultiTaskLoss(coarse_weight=coarse_weight, gamma=gamma),
            accuracy=accuracy, f1_score=f1_calc,
            save_checkpoint=save_checkpoint,
            device=device, epochs=epochs, output_dir=output_dir,
            class_map=PERSON_ACTION_TO_IDX, print_perclass=True,
        )
        trainer.train()
    """
        
    def compute_loss(self, outputs, y):
        """
         MultiTaskLoss returns (total, action_loss, motion_loss, action_targets).
         We keep the extra losses on self so print_epoch can show them.
        """

        total, a_loss, m_loss, action_targets = self.loss_fn(outputs, y)
        motion_logits, action_logits = outputs
        preds = action_logits.argmax(dim=1)
        
        # sent extra loss
        extras = {
            "action": a_loss.item(),
            "motion": m_loss.item()
        }
        return total, preds, action_targets, extras

        
    def compute_metrics(self, all_preds: list, all_targets: list) -> float:
        return self.f1_score(all_targets, all_preds)

    def print_epoch(self, epoch: int, lr: float, train: dict, val: dict):
        print(f"\nEpoch [{epoch}] | lr: {lr:.7f}")
        
        t_act, t_mot = train['extras']['action'], train['extras']['motion']
        v_act, v_mot = val['extras']['action'], val['extras']['motion']
        
        print(f"TRAIN → total: {train['loss']:.3f} | action: {t_act:.3f} | motion: {t_mot:.3f} | F1: {train['metric']:.3f}")
        print(f"VAL   → total: {val['loss']:.3f}   | action: {v_act:.3f} | motion: {v_mot:.3f} | F1: {val['metric']:.3f}")