import torch
from sklearn.metrics import f1_score

def accuracy(all_targets, all_preds):
    """
    all_targets: list[Tensor]  shape (B,)
    all_preds:   list[Tensor]  shape (B,)
    """
    y_true = torch.cat(all_targets)
    y_pred = torch.cat(all_preds)

    correct = (y_true == y_pred).sum().item()
    total = y_true.numel()

    return correct / total
    
  
def f1_calc(all_targets,all_preds):
    y_true = torch.cat(all_targets).cpu().numpy()
    y_pred = torch.cat(all_preds).cpu().numpy()
    return f1_score(y_true, y_pred, average="macro")
