import os
import argparse
import torch
import pytorch_lightning as pl
import pandas as pd
import json
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint

from binding_data import PairDataset, CollaterLBA
from models.protein_model import BindingAffinityPredictor
from utils.plots import plot_regression_metrics
import numpy as np

def draw_r2(predictions, ground_truths, folder_name, fig_name):
    r2 = r2_score(ground_truths, predictions)
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, ground_truths, s=50, alpha=0.7)
    plt.plot([min(predictions), max(predictions)], [min(predictions), max(predictions)], 'k--', lw=2)
    plt.xlabel("Predictions")
    plt.ylabel("Ground Truths")
    plt.title(f"R2 Score: {r2:.3f}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    plt.savefig(os.path.join(folder_name, fig_name), transparent=True, bbox_inches='tight')
    
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--data", type=str, default="kiba")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=str or int, default="cuda")
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints/")
    args = parser.parse_args()

    # DataLoader setup for testing
    root = Path(args.root) / args.data
    csv_file = root / "full.csv"
    df = pd.read_csv(csv_file)
    test_fold = json.load(open(root / "folds/test_fold_setting1.txt"))
    df_test = df[df.index.isin(test_fold)]
    print(f"Test DF:\n{df_test}")
    
    test_dataset = PairDataset(root=root, df=df_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=CollaterLBA(64))

    # Model initialization
    model = BindingAffinityPredictor()
    checkpoint_path = Path(args.chkpt_path) / f"binding_affinity_{args.data}_fold_{args.fold_idx}.ckpt"
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])

    # Trainer configuration
    trainer = pl.Trainer(accelerator=args.device)

    # Testing the model
    trainer.test(model, dataloaders=test_loader)

    # Plotting results (They are already numpy arrays) - Also need to put into dataframe
    predictions = np.concatenate(model.predictions).ravel()
    targets = np.concatenate(model.targets).ravel()

    HERE = Path(__file__)
    partial_plot_save_path = HERE.parent / "plots" / f"binding_affinity_metrics_fold_{args.fold_idx}.png"
    plot_objects = plot_regression_metrics(predictions, targets, f"Binding Affinity Model Metrics, Fold: {args.fold_idx},",
                                           filepath=partial_plot_save_path, with_rm2=True, with_ci=True)