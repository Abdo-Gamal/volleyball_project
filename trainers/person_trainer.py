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

    # ── HOOK 2: multitask loss ────────────────────────────────────────────────
    def compute_loss(self, outputs, y):
        """
        MultiTaskLoss returns (total, action_loss, motion_loss, action_targets).
        We keep the extra losses on self so print_epoch can show them.
        """
        total, a_loss, m_loss, action_targets = self.loss_fn(outputs, y)

        # Store for print_epoch — reset each batch is fine, we only print once
        self._last_action_loss = a_loss.item()
        self._last_motion_loss = m_loss.item()

        motion_logits, action_logits = outputs
        preds = action_logits.argmax(dim=1)

        return total, preds, action_targets


    # ── HOOK 3: checkpoint by F1, not accuracy ────────────────────────────────
    def compute_metrics(self, all_preds: list, all_targets: list) -> float:
        return self.f1_score(all_targets, all_preds)

    # ── HOOK 4: print 3 losses ────────────────────────────────────────────────
    def print_epoch(self, epoch: int, lr: float,
                    train: dict, val: dict):
        print(f"\nEpoch [{epoch}] | lr: {lr:.7f}")
        print(f"TRAIN → loss: {train['loss']:.3f} | F1: {train['metric']:.3f}")
        print(f"VAL   → loss: {val['loss']:.3f}   | F1: {val['metric']:.3f}")

    # ── HOOK 5: checkpoint filename ───────────────────────────────────────────
    def _checkpoint_name(self) -> str:
        return "PersonModel_best.pth"

    # ─────────────────────────────────────────────────────────────────────────
    #  _run_epoch override: we need per-epoch loss accumulators for
    #  action and motion separately, which BaseTrainer doesn't track.
    #  We override only to add that tracking; the rest of the logic
    #  calls super()._run_epoch() internally via the same pattern.
    #
    #  Simpler approach used here: accumulate action/motion loss totals
    #  inside compute_loss via instance variables, then average in print_epoch.
    #  This avoids re-implementing the full loop.
    # ─────────────────────────────────────────────────────────────────────────

    def train(self):
        """
        Extends BaseTrainer.train() to accumulate per-epoch action/motion
        loss totals for the epoch-level print.
        """
        for epoch in range(self.epochs):
            # Reset accumulators
            self._epoch_action_loss = 0.0
            self._epoch_motion_loss = 0.0
            #self._epoch_batches     = 0

            train_result = self._run_epoch_person(
                self.train_loader, training=True
            )
            val_result = self._run_epoch_person(
                self.val_loader, training=False
            )

            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.train_losses.append(train_result["loss"])
            self.val_metrics.append(val_result["metric"])
            self.lr_history.append(lr)

            print(f"\nEpoch [{epoch}] | lr: {lr:.7f}")
            print(
                f"TRAIN → total: {train_result['loss']:.3f} | "
                f"action: {train_result['action_loss']:.3f} | "
                f"motion: {train_result['motion_loss']:.3f}"
            )
            print(
                f"VAL   → total: {val_result['loss']:.3f}   | "
                f"action: {val_result['action_loss']:.3f} | "
                f"motion: {val_result['motion_loss']:.3f}"
            )
            print(
                f"TRAIN F1: {train_result['metric']:.3f} | "
                f"VAL F1:   {val_result['metric']:.3f}"
            )

            if val_result["metric"] >= self.best_metric:
                self.best_metric = val_result["metric"]
                import os
                path = os.path.join(self.output_dir, self._checkpoint_name())
                self.save_checkpoint(
                    self.model, self.optimizer,
                    epoch, self.best_metric, path
                )
                print(
                    f"==> New Best F1: {self.best_metric:.4f}"
                    f"  saved → {path}"
                )

    def _run_epoch_person(self, loader, training: bool) -> dict:
        """
        Extended _run_epoch that tracks action and motion losses separately.
        """
        import torch
        from torch import amp

        self.model.train(training)

        total_loss    = 0.0
        action_loss_t = 0.0
        motion_loss_t = 0.0
        batch_count   = 0
        all_preds     = []
        all_targets   = []
        class_count   = torch.zeros(len(self.class_map), device=self.device)
        class_correct = torch.zeros(len(self.class_map), device=self.device)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x, y in loader:
                if y is None or x is None or x.size(0) <= 1:
                    continue

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=self.use_amp):
                    motion_logits, action_logits = self.model(x)
                    total, a_loss, m_loss, action_targets = self.loss_fn(
                        (motion_logits, action_logits), y
                    )

                if training:
                    self.scaler.scale(total).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                preds = action_logits.argmax(dim=1)
                all_preds.append(preds.detach())
                all_targets.append(action_targets.detach())

                total_loss    += total.item()
                action_loss_t += a_loss.item()
                motion_loss_t += m_loss.item()
                batch_count   += 1

                if self.print_perclass:
                    for name, cls in self.class_map.items():
                        mask = (action_targets == cls)
                        class_count[cls]   += mask.sum()
                        class_correct[cls] += (preds[mask] == cls).sum()

        if self.print_perclass:
            phase = "TRAIN" if training else "VAL"
            self._print_perclass(class_count, class_correct, phase)

        n = max(batch_count, 1)
        return {
            "loss":        total_loss    / n,
            "action_loss": action_loss_t / n,
            "motion_loss": motion_loss_t / n,
            "metric":      self.f1_score(all_targets, all_preds),
        }
    