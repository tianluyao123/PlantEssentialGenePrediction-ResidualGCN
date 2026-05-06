import os

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from torch_geometric.loader import DataLoader


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    if len(y_true) == 0:
        return {"SN": 0.0, "SP": 0.0, "ACC": 0.0, "MCC": 0.0, "AUC": 0.0,
                "Youden_J": 0.0, "G_mean": 0.0, "AUC_PR": 0.0}

    if len(np.unique(y_true)) < 2:
        AUC = AUC_PR = 0.0
    else:
        try:
            AUC = roc_auc_score(y_true, y_prob[:, 1])
            AUC_PR = average_precision_score(y_true, y_prob[:, 1])
        except:
            AUC = AUC_PR = 0.0

    TP = np.sum((y_true == 1) & (y_pred == 1)).astype(np.float64)
    TN = np.sum((y_true == 0) & (y_pred == 0)).astype(np.float64)
    FP = np.sum((y_true == 0) & (y_pred == 1)).astype(np.float64)
    FN = np.sum((y_true == 1) & (y_pred == 0)).astype(np.float64)

    SN = TP / (TP + FN) if (TP + FN) > 1e-8 else 0.0
    SP = TN / (TN + FP) if (TN + FP) > 1e-8 else 0.0
    ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 1e-8 else 0.0
    PPV = TP / (TP + FP) if (TP + FP) > 1e-8 else 0.0

    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = (TP * TN - FP * FN) / denominator if denominator > 1e-8 else 0.0
    MCC = max(min(MCC, 1.0), -1.0)

    Youden_J = SN + SP - 1
    G_mean = (SN * SP) ** 0.5 if (SN > 0 and SP > 0) else 0.0
    F1 = 2 * (PPV * SN) / (PPV + SN) if (PPV + SN) > 1e-8 else 0.0

    return {
        "SN": round(SN, 4), "SP": round(SP, 4), "ACC": round(ACC, 4),
        "PPV": round(PPV, 4), "F1": round(F1, 4),
        "MCC": round(MCC, 4), "AUC": round(AUC, 4), "AUC_PR": round(AUC_PR, 4),
        "Youden_J": round(Youden_J, 4), "G_mean": round(G_mean, 4),
        "TP": int(TP), "TN": int(TN), "FP": int(FP), "FN": int(FN)
    }


def evaluate_model(model: nn.Module, loader: DataLoader, config, set_name: str,
                   save_report: bool = True, fold: int = -1, param_combo: str = "") -> dict:
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.DEVICE, non_blocking=True)
            try:
                if batch.x_p.dtype != torch.float32:
                    batch.x_p = batch.x_p.float()
                if batch.x_f.dtype != torch.float32:
                    batch.x_f = batch.x_f.float()

                logits = model(batch)
                y_true = batch.y.view(-1).cpu().numpy()
                y_pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_prob = F.softmax(logits, dim=1).cpu().numpy()

                all_true.extend(y_true)
                all_pred.extend(y_pred)
                all_prob.extend(y_prob)
            except Exception as e:
                print(f"Warning: Evaluation error: {e}")
                continue

    if len(all_true) == 0:
        return {"SN": 0.0, "SP": 0.0, "ACC": 0.0, "MCC": 0.0, "AUC": 0.0, "Youden_J": 0.0, "G_mean": 0.0}

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)
    metrics = calculate_metrics(all_true, all_pred, all_prob)

    sn_sp_gap = abs(metrics['SN'] - metrics['SP'])
    if sn_sp_gap > 0.15:
        print(f"  Warning: {set_name} set SN-SP gap is too large: |{metrics['SN']:.3f} - {metrics['SP']:.3f}| = {sn_sp_gap:.3f}")

    if save_report:
        try:
            report = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            fname = f"{param_combo}_{set_name}_report.csv" if param_combo else f"fold_{fold}_{set_name}_report.csv"
            report_df.to_csv(os.path.join(config.SAVE_BASE_PATH, fname), index=True, encoding="utf-8-sig")
        except Exception as e:
            print(f"Warning: Failed to save report: {e}")

    print(f"  - {set_name} set: SN={metrics['SN']:.4f}, SP={metrics['SP']:.4f}, "
          f"YoudenJ={metrics['Youden_J']:.4f}, Gmean={metrics['G_mean']:.4f}, AUC={metrics['AUC']:.4f}")
    return metrics


def evaluate_youden_on_loader(model: nn.Module, loader: DataLoader, config) -> tuple[float, float, float]:

    model.eval()
    tp = tn = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(config.DEVICE, non_blocking=True)
            if hasattr(batch, 'x_p') and batch.x_p.dtype != torch.float32:
                batch.x_p = batch.x_p.float()
            if hasattr(batch, 'x_f') and batch.x_f.dtype != torch.float32:
                batch.x_f = batch.x_f.float()

            logits = model(batch)
            preds = torch.argmax(logits, dim=1)
            targets = batch.y.view(-1)

            tp += int(((targets == 1) & (preds == 1)).sum().item())
            tn += int(((targets == 0) & (preds == 0)).sum().item())
            fp += int(((targets == 0) & (preds == 1)).sum().item())
            fn += int(((targets == 1) & (preds == 0)).sum().item())

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    youden_j = sn + sp - 1.0
    return float(youden_j), float(sn), float(sp)
