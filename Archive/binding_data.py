import os
import torch
from torch_geometric.data import Data, Batch
import selfies as sf
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset
import pickle
import numpy as np
from pathlib import Path
from utils.database_id_mapping import DataBaseIDMapper
from utils.protein_utils import featurize_as_graph
from cadd_lead_optimisation.utils.helpers.pdb import fetch_and_save_pdb_file
import atom3d.util.formats as atom3d_formats
import warnings

# Disable warnings and set seed for reproducibility
warnings.filterwarnings('ignore')

class PairDataset(Dataset):
    def __init__(self, root=None, df=None):
        super().__init__()
        self.df = df
        self.root = Path(root) if root else None
        self.mapper = DataBaseIDMapper()  # Initialize the database ID mapper

        # Assuming the existence of ligand_to_graph.pkl and ligand_to_ecfp.pkl files
        self.ligand_graphs = pickle.load(open(self.root / "ligand_to_graph.pkl", "rb"))
        self.ligand_mps = pickle.load(open(self.root / "ligand_to_ecfp.pkl", "rb"))

    def __len__(self):
        return len(self.df)

    def prepare_protein_data(self, protein_name):
        protein_3d_path = self.root / "prot_3d"
        protein_pdb_path = protein_3d_path / f"{protein_name}.pdb"

        # Fetch and save the protein file if it does not exist
        if not protein_pdb_path.exists():
            mapped_protein_id = self.map_protein_id(protein_name)
            fetch_and_save_pdb_file(mapped_protein_id, str(protein_pdb_path))  # Ensure path is a string for compatibility

        # Read the protein file and featurize it as a graph
        protein_df = atom3d_formats.bp_to_df(atom3d_formats.read_pdb(str(protein_pdb_path)))
        return featurize_as_graph(protein_df)

    def map_protein_id(self, protein_name):
        # Convert UniProt ID to PDB ID
        conversion_dict = self.mapper.map_ids("UniProtKB_AC-ID", "PDB", [protein_name])
        mapped_id = conversion_dict.get(protein_name)
        if not mapped_id:
            raise ValueError(f"No PDB ID found for UniProt ID {protein_name}")
        return mapped_id

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        protein_name = row["protein"]
        ligand = row["ligand"]

        # Load or prepare the protein graph
        protein_file_path = self.root / "res_graph" / f"{protein_name}.pdb.pt"
        if not protein_file_path.exists():
            protein_graph = self.prepare_protein_data(protein_name)
        else:
            protein_graph = torch.load(protein_file_path, map_location="cpu")

        # Check if label exists in the row and set it to the protein graph if it does
        if 'label' in row and row['label'] is not None:
            label = row['label']
            protein_graph.y = torch.tensor([label], dtype=torch.float)
        else:
            # Optionally handle the case where label does not exist
            protein_graph.y = None  # Or any placeholder you deem appropriate

        # Handle ligand graph creation
        ligand_graph_data = self.ligand_graphs[ligand]
        ligand_graph = Data(
            x=torch.tensor(np.asarray(ligand_graph_data[0], dtype=np.float32), dtype=torch.float),
            edge_index=torch.tensor(np.asarray(ligand_graph_data[1], dtype=np.int64), dtype=torch.long).t(),
            edge_attr=torch.tensor(np.asarray(ligand_graph_data[2], dtype=np.float32), dtype=torch.float))

        ligand_mp = torch.tensor(self.ligand_mps[ligand], dtype=torch.float)

        return protein_graph, ligand_graph, ligand_mp

class CollaterLBA(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def collate(self, data_list):
        if len(data_list) != self.batch_size:
            bs = len(data_list)
        else:
            bs = self.batch_size
        batch_1 = Batch.from_data_list([d[0] for d in data_list])
        batch_2 = Batch.from_data_list([d[1] for d in data_list])
        mp = torch.stack([d[2] for d in data_list])
        return batch_1, batch_2, mp
    
    def adjust_graph_indices(self, graph, bs):
        total_n = 0
        for i in range(bs-1):
            n_nodes = graph.num_nodes[i].item()
            total_n += n_nodes
            #graph.ca_idx[i+1] += total_n
        return graph

    def __call__(self, batch):
        return self.collate(batch)


class TargetDataset(Dataset):
    def __init__(self, root, csv_file, symbol_to_idx, 
                idx_to_symbol, max_len, protein_set, transform = None, test = False,
                ):
        super().__init__()
        self.root = root
        self.csv_file = csv_file
        self.symbol_to_idx = symbol_to_idx
        self.idx_to_symbol = idx_to_symbol 
        self.max_len = max_len
        self.transform = transform 
        self.test = test
        self.df = pd.read_csv(self.csv_file)
        self.df = self.df[self.df.protein.isin(protein_set)]
 
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        items = self.df.iloc[idx]
        protein_name = items['protein']
        ligand = items['ligand']
        label = items["label"]
        name = protein_name + "_" + ligand
        s = sf.encoder(ligand)
        encoding = [self.symbol_to_idx[symbol] for symbol in sf.split_selfies(s)]
        if len(encoding) < self.max_len:
            ligand_tensor = torch.tensor(encoding + [self.symbol_to_idx['[nop]'] for i in range(self.max_len - len(encoding))])
        else:
            ligand_tensor = torch.tensor(encoding)
        protein_file = os.path.join(self.root, f"{protein_name}.pdb.pt")
        protein_graph = torch.load(protein_file, map_location="cpu")
        protein_graph.y = label 
        
        if self.test:
            return ligand_tensor, protein_graph, int(label), ligand, protein_name
        return ligand_tensor, protein_graph, int(label)
    
class VAECollate():
    def __init__(self, test = False):
        self.test = test

    def __call__(self, data_list):
        ligand_tensor = torch.stack([d[0] for d in data_list], dim = 0)
        protein_graph = Batch.from_data_list([d[1] for d in data_list])
        return ligand_tensor, protein_graph