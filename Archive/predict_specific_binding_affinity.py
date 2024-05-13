from torch_geometric.data import Batch
import copy
from utils.mol_utils import one_hot_to_smiles, smiles_to_one_hot
from utils.database_id_mapping import DataBaseIDMapper
from utils.protein_utils import featurize_as_graph
import atom3d.util.formats as atom3d_formats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import math

import os
import argparse
import torch
import pytorch_lightning as pl
import json
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from binding_data import PairDataset, CollaterLBA
from models.protein_model import BindingAffinityPredictor
from utils.plots import plot_regression_metrics
import numpy as np

import pandas as pd
from rdkit import Chem
from cadd_lead_optimisation.utils.helpers.rdkit import calculate_druglikeness
from cadd_lead_optimisation.utils.helpers.pdb import fetch_and_save_pdb_file
from pathlib import Path

from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SpecificBindingAffinityPredictor:
    def __init__(self, args):
        self.args = args
        self.HERE = Path(__file__).resolve().parent
        self.root = Path(self.args.root) / self.args.data
        self.csv_file = self.root / "full.csv"
        self.partial_plot_save_path = self.HERE / "plots" / f"binding_affinity_metrics_fold_{self.args.fold_idx}.png"
        self.test_fold_path = self.root / "folds/test_fold_setting1.txt"
        self.checkpoint_path = Path(self.args.chkpt_path) / f"binding_affinity_{self.args.data}_fold_{self.args.fold_idx}.ckpt"
        self.ligand_smiles = self.args.ligand_smiles
        self.bap = None

        self.set_random_seed()
        self.create_template_df()
        self.setup_dataloaders()
        self.load_binding_affinity_model()
        self.predict_binding_affinities_model()
        self.get_results()

    def set_random_seed(self, seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def load_binding_affinity_model(self):
        self.bap = BindingAffinityPredictor()
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        self.bap.load_state_dict(checkpoint['state_dict'])
        self.bap.eval()

    def create_template_df(self):
        df_values = {
            "ligand": self.ligand_smiles,
            "protein": [self.args.protein_id] * len(self.ligand_smiles),
        }
        self.df_test = pd.DataFrame(df_values)

    def setup_dataloaders(self):
        self.test_dataset = PairDataset(root=self.root, df=self.df_test)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=CollaterLBA(64))

    def predict_binding_affinities_model(self):
        trainer = pl.Trainer(accelerator=self.args.device, devices=1 if self.args.device == 'cuda' else None)
        trainer.test(model=self.bap, dataloaders=self.test_loader)

    def get_results(self):
        self.predictions = np.concatenate(self.bap.predictions).ravel()
        print(f"############################ Results ############################")
        print(f"\nPREDICTIONS: {self.predictions}\nLEN PREDICTIONS: {len(self.predictions)}\nLEN INPUTS: {len(self.ligand_smiles)}")

def parse_arguments():
    """Parse command-line arguments."""
    # Need to add extra part to check if input database type is that of the allowed ones for pdb api
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--data", type=str, default="kiba") # Data for LIGANDS
    parser.add_argument("--fold_idx", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ligand_smiles", nargs="+", type=str) # List of input ligands
    parser.add_argument("--protein_id", type=str, default="P21802") # Protein to GEN BINDING AFFINITES TO ALL DATA
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints/")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    binding_affinity_predictor = SpecificBindingAffinityPredictor(ligand_smiles, args)
