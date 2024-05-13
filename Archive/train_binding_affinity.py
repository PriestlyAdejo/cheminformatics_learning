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

# Function definitions
def set_random_seed(seed_value=12345):
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_new_file_path(base_path, file_name):
    counter = 1
    new_file_path = os.path.join(base_path, file_name)
    while os.path.exists(new_file_path):
        new_file_path = Path(base_path) / f"{file_name.split('.')[0]}_{counter}.{file_name.split('.')[1]}"
        counter += 1
    return new_file_path

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

# Main script
if __name__ == "__main__":
    set_random_seed()

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--data", type=str, default="kiba")
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=str|int, default="cuda")
    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints/")
    args = parser.parse_args()

    # Dataset preparation
    root = Path(args.root) / args.data
    csv_file = root / "full.csv"
    df = pd.read_csv(csv_file)
    test_fold = json.load(open(root / "folds/test_fold_setting1.txt"))
    folds = json.load(open(root / "folds/train_fold_setting1.txt"))
    val_fold = folds[args.fold_idx]
    df_train = df[~df.index.isin(test_fold) & ~df.index.isin(val_fold)]
    df_val = df[df.index.isin(val_fold)]
    df_test = df[df.index.isin(test_fold)]
    train_dataset = PairDataset(root=root, df=df_train)
    val_dataset = PairDataset(root=root, df=df_val)
    test_dataset = PairDataset(root=root, df=df_test)

    # DataLoader setup
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=CollaterLBA(batch_size))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=CollaterLBA(batch_size))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=CollaterLBA(batch_size))

    # Model initialization
    model = BindingAffinityPredictor()
    checkpoint_callback = ModelCheckpoint(dirpath=args.chkpt_path, save_top_k=1, monitor="val_loss", mode="min", filename=f"binding_affinity_{args.data}_fold_{args.fold_idx}")

    # Trainer configuration
    trainer = pl.Trainer(accelerator=args.device, max_epochs=args.num_epoch, logger=pl.loggers.CSVLogger('logs'), callbacks=[checkpoint_callback], gradient_clip_val=0.5, accumulate_grad_batches=2, precision=16)

    # Train the model
    trainer.fit(model, train_loader, val_loader)