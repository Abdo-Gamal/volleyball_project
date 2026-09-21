import os
import torch
import torch.distributed as dist          # [CHANGED 0] needed for the 3 helper functions at the bottom
from torch import amp

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark     = True

# =========================================================================
# 1. BaseTrainer  (your file, with 5 small changes marked  [CHANGED n])
# =========================================================================
class BaseTrainer:
    def __init__(self, model, train_loader, val_loader,
                 optimizer, scheduler, loss_fn,
                 accuracy, f1_score, save_checkpoint,
                 device, epochs, output_dir, class_map,
                 print_perclass: bool = False,
                 checkpoint_name: str = "best_model.pth"): 

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
        self.checkpoint_name = checkpoint_name

        self.use_amp = (device.type == "cuda")
        self.scaler  = amp.GradScaler(enabled=self.use_amp)

        # [CHANGED 1] the master is GLOBAL rank 0 (RANK), not LOCAL_RANK.
        #             On one node they are the same; on two nodes LOCAL_RANK==0 is true twice.
        self.is_master = int(os.environ.get("RANK", 0)) == 0

        self.best_Accuracy  = 0.0
        self.train_losses = []
        self.val_Accuracys  = []
        self.lr_history   = []

    def train(self):
        for epoch in range(self.epochs):
            
            # different shuffling in every epoch
            if hasattr(self.train_loader, 'sampler') and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            train_result = self._run_epoch(self.train_loader, training=True)
            val_result   = self._run_epoch(self.val_loader,   training=False)

            self.scheduler.step()

            # print / save only on the master
            if self.is_master:
                lr = self.optimizer.param_groups[0]["lr"]
                self.train_losses.append(train_result["loss"])
                self.val_Accuracys.append(val_result["f1"])
                self.lr_history.append(lr)

                self.print_epoch(epoch, lr, train_result, val_result)

                if val_result["f1"] >= self.best_Accuracy:
                    self.best_Accuracy = val_result["f1"]
                    path = os.path.join(self.output_dir, self.checkpoint_name)
                    
                    # save the model WITHOUT the DDP wrapper
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
                    
                    self.save_checkpoint(
                        model_to_save, self.optimizer,
                        epoch, self.best_Accuracy, path
                    )
                    print(f"==> New Best: {self.best_Accuracy:.4f}  saved → {path}")
                    
    def _run_epoch(self, loader, training: bool) -> dict:
        self.model.train(training)

        total_loss    = 0.0
        batch_count   = 0
        all_preds     = []
        all_targets   = []
        extra_losses_totals = {} #extra loss for multitask 

        class_count   = torch.zeros(len(self.class_map), device=self.device)
        class_correct = torch.zeros(len(self.class_map), device=self.device)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for x, y in loader:
                if y is None or y.size(0) <= 1:
                    continue

                x = self.move_input(x)
                y = y.to(self.device, non_blocking=True)

                if training:
                    self.optimizer.zero_grad(set_to_none=True)

                with amp.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(x)
                    # extras empty dictionary by default if not make multitask 
                    loss, preds, targets, extras = self.compute_loss(outputs, y)

                if training:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                all_preds.append(preds.detach())
                all_targets.append(targets.detach())
                total_loss  += loss.item()
                batch_count += 1
                
                for k, v in extras.items():
                    extra_losses_totals[k] = extra_losses_totals.get(k, 0.0) + v

                if self.print_perclass:
                    for name, cls in self.class_map.items():
                        mask = (targets == cls)
                        class_count[cls]   += mask.sum()
                        class_correct[cls] += (preds[mask] == cls).sum()

        # [CHANGED 2] per-class table: add the counters of ALL GPUs, print on the master only
        #             (before: every GPU printed its own table -> 4 mixed-up copies, each on 1/4 of the data)
        if self.print_perclass:
            class_count   = self._sum_over_gpus(class_count)
            class_correct = self._sum_over_gpus(class_correct)
            if self.is_master:
                phase = "TRAIN" if training else "VAL"
                self._print_perclass(class_count, class_correct, phase)

        # [CHANGED 3] accuracy / f1 on the predictions of ALL GPUs (before: only this GPU's quarter)
        all_preds   = [self._gather_all(all_preds)]
        all_targets = [self._gather_all(all_targets)]

        Accuracy = self.compute_Accuracys(all_preds, all_targets)
        f1 = self.compute_f1(all_preds, all_targets)

        # [CHANGED 4] mean loss over ALL GPUs
        stats = self._sum_over_gpus(torch.tensor([total_loss, float(batch_count)],
                                                 dtype=torch.float64, device=self.device))
        mean_loss = stats[0].item() / max(stats[1].item(), 1.0)

        n = max(batch_count, 1)      # extras stay local (baseline5 has none)

        return {
            "loss":    mean_loss,
            "Accuracy":  Accuracy,
            "f1": f1,
            "preds":   all_preds,
            "targets": all_targets,
            "extras":  {k: v / n for k, v in extra_losses_totals.items()}
        }

    # =========================================================================
    #  [CHANGED 5] helpers that talk to the other GPUs
    # =========================================================================
    def _sum_over_gpus(self, t):
        """sum a tensor over all GPUs (on the CPU, which is what gloo handles best)"""
        if dist.is_available() and dist.is_initialized():
            t = t.detach().cpu()
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t

    def _gather_all(self, tensors):
        """concatenate the predictions of every GPU into one tensor, identical on all GPUs"""
        t = torch.cat(tensors).cpu() if len(tensors) else torch.zeros(0, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            parts = [None] * dist.get_world_size()
            dist.all_gather_object(parts, t)
            t = torch.cat(parts)
        return t.to(self.device)

    # =========================================================================
    #  HOOKS  (unchanged)
    # =========================================================================
    def move_input(self, x):
        return x.to(self.device, non_blocking=True)

    def compute_loss(self, outputs, y):
        loss  = self.loss_fn(outputs, y)
        preds = outputs.argmax(dim=1)
        return loss, preds, y, {} 

    def compute_Accuracys(self, all_preds: list, all_targets: list) -> float:
        return self.accuracy(all_targets, all_preds)

    def compute_f1(self, all_preds: list, all_targets: list) -> float:
        return self.f1_score(all_targets, all_preds)

    def print_epoch(self, epoch: int, lr: float, train: dict, val: dict):
        print(f"\nEpoch [{epoch}] | lr: {lr:.7f}")
        print(f"TRAIN → loss: {train['loss']:.3f} | acc: {train['Accuracy']:.3f} | f1: {train['f1']:.3f}")
        print(f"VAL   → loss: {val['loss']:.3f}   | acc: {val['Accuracy']:.3f}   | f1: {val['f1']:.3f}")


    def _print_perclass(self, count: torch.Tensor, correct: torch.Tensor, phase: str):
        print(f"\n--- {phase} Per-Class ---")
        for name, i in self.class_map.items():
            acc = 100.0 * correct[i] / (count[i] + 1e-6)
            print(f"  {name:12s}: n={int(count[i]):5d} | acc={acc:.1f}%")