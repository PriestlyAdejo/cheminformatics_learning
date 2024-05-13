import os
import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import warnings

from binding_data import TargetDataset, VAECollate
from models.conditional_vae import ThreeD_Conditional_VAE
from pathlib import Path

# Function to create a new file name if the original exists
def create_new_file_path(base_path, file_name):
    counter = 1
    new_file_path = os.path.join(base_path, file_name)
    while os.path.exists(new_file_path):
        new_file_path = Path(base_path) / f"{file_name.split('.')[0]}_{counter}.{file_name.split('.')[1]}"
        counter += 1
    return new_file_path

# Suppress warnings for cleaner output and set a random seed for reproducibility
warnings.filterwarnings('ignore')
torch.manual_seed(1)

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./data/")
parser.add_argument("--data", type=str, default="kiba")
parser.add_argument("--fold_idx", type=int, default=0)
parser.add_argument("--device", type=str or int, default=0)
parser.add_argument("--num_epoch", type=int, default=30)
parser.add_argument("--chkpt_path", type=str, default="./checkpoints/")
args = parser.parse_args()

# Prepare dataset paths
csv_file = os.path.join(args.root, args.data, "filter.csv")
res_graph_path = os.path.join(args.root, args.data, "res_graph")

# Load symbol to index mappings
with open("symbol_to_idx.json", "r") as f:
    symbol_to_idx = json.load(f)
with open("idx_to_symbol.json", "r") as f:
    idx_to_symbol = json.load(f)

# Prepare protein dataset
protein_files = os.listdir(res_graph_path)
protein = sorted(x.replace(".pdb.pt", "") for x in protein_files)
num_train = int(len(protein) * 0.9)
train_protein, test_protein = protein[:num_train], protein[num_train:]

# Initialize datasets
train_dataset = TargetDataset(
    csv_file=csv_file, root=res_graph_path, symbol_to_idx=symbol_to_idx,
    idx_to_symbol=idx_to_symbol, max_len=72, protein_set=train_protein
)
test_dataset = TargetDataset(
    csv_file=csv_file, root=res_graph_path, symbol_to_idx=symbol_to_idx,
    idx_to_symbol=idx_to_symbol, max_len=72, protein_set=test_protein
)

# DataLoader setup
batch_size = 64
num_workers = 4
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, collate_fn=VAECollate(False))

# Initialize model and training
vae = ThreeD_Conditional_VAE(
    max_len=72, vocab_len=108, latent_dim=1024, embedding_dim=128,
    condition_dim=128, checkpoint_path="vae.pt", freeze=True
)
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor="val_loss",
    mode="min",
    dirpath=args.chkpt_path,
    filename=f"conditional_vae_{args.data}_fold_{args.fold_idx}",
    every_n_epochs=1
)

# Training configuration
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=args.num_epoch,
    logger=pl.loggers.CSVLogger('logs'),
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    accumulate_grad_batches=2,
    precision=16 if True else 32  # Use 16-bit precision for training
)

# Check if the checkpoint directory exists. If not, create it.
if not os.path.exists(args.chkpt_path):
    os.makedirs(args.chkpt_path)

# Train the model
print('Training Variational Auto-Encoder...')
trainer.fit(vae, loader)

# Save the trained model
new_model_save_path = create_new_file_path(args.chkpt_path, "conditional_vae.pt")
print(f'Saving VAE Model to {new_model_save_path}...')
torch.save(vae.state_dict(), new_model_save_path)
