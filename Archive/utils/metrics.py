import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score,
    auc, accuracy_score, matthews_corrcoef, mean_squared_error, mean_absolute_error,
    median_absolute_error, mean_absolute_percentage_error, max_error, r2_score
)
from scipy.stats import pearsonr

# User defined values for getting plots
def get_cindex(y, p):
    assert len(y) == len(p), "Lengths of true values and predictions must be equal."

    n = len(y)
    y_matrix = np.tile(y, (n, 1))
    p_matrix = np.tile(p, (n, 1))

    # Create a mask for all unique pairs
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    # Compare all unique pairs of true values and predictions
    y_matrix = y_matrix.T - y_matrix
    p_matrix = p_matrix.T - p_matrix

    concordant = np.sum((y_matrix > 0) & (p_matrix > 0) & mask)
    discordant = np.sum((y_matrix > 0) & (p_matrix < 0) & mask)
    ties = np.sum((y_matrix > 0) & (p_matrix == 0) & mask)

    c_index = (concordant + 0.5 * ties) / (concordant + discordant + ties) if (concordant + discordant + ties) > 0 else 0
    return c_index

def calculate_mean_diffs(y_obs, y_pred):
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    y_obs_diff = y_obs - y_obs_mean
    y_pred_diff = y_pred - y_pred_mean
    return y_obs_diff, y_pred_diff, y_obs_mean

def r_squared_error(y_obs, y_pred):
    y_obs_diff, y_pred_diff, _ = calculate_mean_diffs(y_obs, y_pred)
    mult = np.sum(y_pred_diff * y_obs_diff) ** 2
    y_obs_sq = np.sum(y_obs_diff ** 2)
    y_pred_sq = np.sum(y_pred_diff ** 2)
    return mult / (y_obs_sq * y_pred_sq)

def squared_error_zero(y_obs, y_pred, y_obs_mean):
    k = np.sum(y_obs * y_pred) / np.sum(y_pred ** 2)
    numerator = np.sum((y_obs - (k * y_pred)) ** 2)
    denominator = np.sum((y_obs - y_obs_mean) ** 2)
    return 1 - (numerator / denominator)

def get_rm2(y_obs, y_pred):
    y_obs_diff, _, y_obs_mean = calculate_mean_diffs(y_obs, y_pred)
    r2 = r_squared_error(y_obs, y_pred)
    r02 = squared_error_zero(y_obs, y_pred, y_obs_mean)
    return r2 * (1 - np.sqrt(abs(r2 ** 2 - r02 ** 2)))

# Putting everything together
def get_metrics_reg(y_true, y_pred, with_rm2=False, with_ci=False):
    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "maxe": max_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "pearsonr": pearsonr(y_true.flatten(), y_pred.flatten())[0]
    }
    if with_rm2:
        metrics["rm2"] = get_rm2(y_true.flatten(), y_pred.flatten())
    if with_ci:
        metrics["ci"] = get_cindex(y_true.flatten(), y_pred.flatten())

    return metrics

def get_metrics_cls(y_true, y_pred, transform=torch.sigmoid, threshold=0.5):
    # Ensure y_pred is a torch tensor and apply transformation if needed
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    y_pred = transform(y_pred) if transform else y_pred

    # Convert y_pred to binary labels based on the threshold
    y_pred_lbl = (y_pred >= threshold).type(torch.float32)

    # Ensure y_true is a torch tensor
    y_true = torch.tensor(y_true, dtype=torch.float32) if not isinstance(y_true, torch.Tensor) else y_true

    # Convert tensors to numpy arrays for metric calculations
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()
    y_pred_lbl_np = y_pred_lbl.detach().cpu().numpy()

    # Calculate classification metrics
    metrics = {
        "f1": f1_score(y_true_np, y_pred_lbl_np), #
        "precision": precision_score(y_true_np, y_pred_lbl_np, zero_division=0), #
        "recall": recall_score(y_true_np, y_pred_lbl_np), #
        "accuracy": accuracy_score(y_true_np, y_pred_lbl_np),
        "mcc": matthews_corrcoef(y_true_np, y_pred_lbl_np)
    }

    # Compute ROC AUC and PR AUC if possible
    try:
        metrics["rocauc"] = roc_auc_score(y_true_np, y_pred_np)
        precision_list, recall_list, _ = precision_recall_curve(y_true_np, y_pred_np)
        metrics["prauc"] = auc(recall_list, precision_list)
    except ValueError:
        metrics["rocauc"] = metrics["prauc"] = np.nan

    return metrics

