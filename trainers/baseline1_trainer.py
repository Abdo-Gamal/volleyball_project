"""
trainers/baseline1_trainer.py
==============================
Baseline 1 uses a single classification loss on whole frames.
This matches exactly what BaseTrainer does by default:
  - move_input   : plain tensor (default)
  - compute_loss : single CrossEntropyLoss (default)
  - compute_metrics : accuracy (default)
  - _checkpoint_name : 'baseline1_best.pth'

So Baseline1Trainer overrides only the checkpoint name.
The entire training loop is inherited from BaseTrainer.

Usage in notebook:
    from trainers.baseline1_trainer import Baseline1Trainer
    trainer = Baseline1Trainer(
        model=model, train_loader=trainloader, val_loader=valloader,
        optimizer=optimizer, scheduler=scheduler,
        loss_fn=nn.CrossEntropyLoss(),
        accuracy=accuracy, f1_score=f1_calc,
        save_checkpoint=save_checkpoint,
        device=device, epochs=epochs, output_dir=output_dir,
        class_map=GROUP_ACTION_TO_IDX, print_perclass=True,
    )
    trainer.train()
"""

from base_trainer import BaseTrainer


class Baseline1Trainer(BaseTrainer):
    """
    Trainer for Baseline 1 (whole-frame group classification).
    Inherits everything from BaseTrainer — no overrides needed
    except the checkpoint filename.
    """

    def _checkpoint_name(self) -> str:
        return "baseline1_best.pth"