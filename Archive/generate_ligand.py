import os
import re
import argparse
import torch
from torch_geometric.data import Batch
import copy
from models.conditional_vae import ThreeD_Conditional_VAE
from utils.mol_utils import one_hot_to_smiles
from utils.database_id_mapping import DataBaseIDMapper
from utils.protein_utils import featurize_as_graph
from utils.chem_evaluator import diversity, uniqueness, novelty, validity
import atom3d.util.formats as atom3d_formats
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import math

import pandas as pd
from rdkit import Chem
from cadd_lead_optimisation.utils.helpers.rdkit import calculate_druglikeness
from cadd_lead_optimisation.utils.helpers.pdb import fetch_and_save_pdb_file
from pathlib import Path

warnings.filterwarnings('ignore')

class MoleculeGenerator:
    def __init__(self, args):
        self.args = args
        self.mapper = DataBaseIDMapper()
        self.vae = None
        self.protein_graph = None
        self.filepaths = {"plots": Path("plots")}
        
        # Running below methods
        self.setup_metrics_performance_and_thresholds()
        self.setup_train_smiles()
        self.load_vae_model()
        self.prepare_protein_data()
        self.generate_molecules()
        self.create_metrics_performance_df()
        self.save_metrics_performance_df()
        self.plot_druggability_and_performance()
        
    def load_vae_model(self):
        self.vae = ThreeD_Conditional_VAE(max_len=72, vocab_len=108, latent_dim=1024,
                                          embedding_dim=128, condition_dim=128, checkpoint_path=None, freeze=False)
        checkpoint_path = os.path.join(self.args.chkpt_path, 'conditional_vae.pt')
        self.vae.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        self.vae.to(self.args.device)
        self.vae.eval()

    def prepare_protein_data(self):
        protein_3d_path = os.path.join(self.args.root, self.args.data, f"prot_3d_for_{self.args.data}")
        protein_pdb_path = os.path.join(protein_3d_path, f"{self.args.protein_name}.pdb")

        # Check if protein data is already available in the directory
        if not os.path.exists(protein_pdb_path):
            # Map protein ID and fetch protein data if not found in the directory
            mapped_protein_id = self.map_protein_id()
            fetch_and_save_pdb_file(mapped_protein_id, protein_pdb_path)

        # Read and process the protein data
        protein_df = atom3d_formats.bp_to_df(atom3d_formats.read_pdb(protein_pdb_path))
        self.protein_graph = featurize_as_graph(protein_df, device=self.args.device)

    def map_protein_id(self):
        conversion_dict = self.mapper.map_ids("UniProtKB_AC-ID", "PDB", [self.args.protein_name])
        mapped_id = conversion_dict.get(self.args.protein_name)
        if not mapped_id:
            raise ValueError(f"No PDB ID found for UniProt ID {self.args.protein_name}")
        return mapped_id

    def generate_molecules(self):
        batch_mol = 500 if self.args.num_mols >= 1000 else 100
        for _ in range(self.args.num_mols // batch_mol):
            cond = Batch.from_data_list([copy.deepcopy(self.protein_graph) for _ in range(batch_mol)])
            cond.to(self.args.device)

            with torch.no_grad():
                x_protein = self.vae.protein_model(
                        (
                            cond.node_s.to(self.args.device),
                            cond.node_v.to(self.args.device)
                        ), 
                        cond.edge_index.to(self.args.device), 
                        (
                            cond.edge_s.to(self.args.device),
                            cond.edge_v.to(self.args.device)
                        ), 
                        cond.seq.to(self.args.device), 
                        cond.batch.to(self.args.device)
                    )
                outs = self.vae.prior_network(x_protein).view(-1, 2, 1024)
                prior_mu, prior_log_var = outs[:, 0, :], outs[:, 1, :]
                prior_std = torch.exp(prior_log_var * 0.5)
                z = torch.normal(mean = prior_mu, std = prior_std)
                one_hots_from_smiles = self.vae.decode(z, None).cpu()
                self.process_generated_molecules(one_hots_from_smiles)

    def process_generated_molecules(self, one_hots_from_smiles):
        
        for one_hot in one_hots_from_smiles:
            smiles = one_hot_to_smiles(one_hot)
            molecule_obj = Chem.MolFromSmiles(smiles)
            if molecule_obj is None:
                continue  # Skip invalid SMILES strings

            properties_dict = calculate_druglikeness(molecule_obj)
            valid = self.is_valid_smiles(properties_dict)

            self.metrics['smiles'].append(smiles)
            self.metrics['is_druggable'].append(valid)

            # Append each property to the corresponding metrics list
            for property_name in self.thresholds.keys():
                property_value = properties_dict.get(property_name, None)
                self.metrics[property_name].append(property_value)
            
        print(f"\nGETTING MODEL PERFORMACE ON GENERATIONS FOR:\n{self.metrics['smiles']}\n")
        # Model Performance
        self.performance = dict(diversity_score = [diversity(self.metrics['smiles'])],
                            uniqueness_score = [uniqueness(self.metrics['smiles'])],
                            novelty_score = [novelty(self.metrics['smiles'], self.train_smiles)], 
                            validity_score = [validity(self.metrics['smiles'])])
        print(f"\nRESUTLS:\n{self.performance}\n")
        
    def is_valid_smiles(self, properties_dict):
        """Check if a molecule's properties are valid based on the set thresholds."""
        for metric, value in properties_dict.items():
            threshold = self.thresholds.get(metric, {})
            if threshold.get('min') is not None and value < threshold['min']:
                return False
            if threshold.get('max') is not None and value > threshold['max']:
                return False
        return True

    def setup_metrics_performance_and_thresholds(self, custom_thresholds=None):
        # Default threshold values
        default_thresholds = {
            'mol_weight': {'min': None, 'max': None},
            'num_H_acceptors': {'min': None, 'max': None},
            'num_H_donors': {'min': None, 'max': None},
            'logp': {'min': None, 'max': None},
            'tpsa': {'min': None, 'max': None},
            'num_rot_bonds': {'min': None, 'max': None},
            'saturation': {'min': None, 'max': None},
            'drug_score_qed': {'min': None, 'max': None},
            'drug_score_lipinski': {'min': None, 'max': None},
            'drug_score_custom': {'min': None, 'max': None},
            'drug_score_total': {'min': 0.7, 'max': None},
        }

        # Update default thresholds with any custom thresholds provided
        if custom_thresholds:
            for key, value in custom_thresholds.items():
                if key in default_thresholds:
                    default_thresholds[key].update(value)

        # Set the thresholds for the class instance
        self.thresholds = default_thresholds

        # Initialize metrics dictionary
        self.metrics = {'smiles': [], 'is_druggable': []}
        self.metrics.update({key: [] for key in self.thresholds})
        self.performance = {}
    
    def setup_train_smiles(self): # Need to modify later to find automatically
        csv_file = os.path.join(self.args.root, self.args.data, "filter.csv")
        df = pd.read_csv(csv_file)
        self.train_smiles = df['ligand'].to_list()
     
    def create_metrics_performance_df(self):
        self.metrics_df = pd.DataFrame(self.metrics)
        self.performance_df = pd.DataFrame(self.performance)
    
    def save_metrics_performance_df(self):
        # Saving metrics
        file_name = 'generated_smiles_data.csv'
        self.metrics_df.to_csv(file_name, index=False)
        print(f"Generated Smiles strings and their benchmarks saved to: {file_name}")
        
        # Saving performance
        file_name = 'model_performance.csv'
        self.performance_df.to_csv(file_name, index=False)
        print(f"Model Performance saved to: {file_name}")

    def plot_druggability_and_performance(self):
        # Ensure directory exists
        os.makedirs(self.filepaths["plots"], exist_ok=True)

        # Plotting the proportion of druggable smiles
        fig, ax = plt.subplots(figsize=(8, 6))
        druggable_counts = self.metrics_df['is_druggable'].value_counts()
        ax.pie(druggable_counts, labels=['Non-Druggable', 'Druggable'], autopct='%1.1f%%', startangle=140)
        ax.set_title('Proportion of Druggable Molecules')
        plt.savefig(os.path.join(self.filepaths["plots"], 'druggable_proportion.png'), bbox_inches='tight')

        # Number of metrics
        num_metrics = len(self.thresholds)
        # Calculate rows and columns for subplots
        num_cols = int(math.ceil(math.sqrt(num_metrics)))
        num_rows = int(math.ceil(num_metrics / num_cols))

        # Create subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
        if num_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Melt dataframe for seaborn
        melted_df = self.metrics_df.melt(id_vars=['is_druggable'], value_vars=list(self.thresholds.keys()))

        # Plot each metric
        for idx, metric in enumerate(self.thresholds.keys()):
            sns.violinplot(x='variable', y='value', hue='is_druggable', 
                           data=melted_df[melted_df['variable'] == metric],
                           split=True, inner='quart', palette='Set2', ax=axes[idx])

            # Set the title, labels, and tick parameters with specific font sizes
            axes[idx].set_title(f'Variations in {metric}', fontsize=18)
            axes[idx].set_xlabel('', fontsize=18)
            axes[idx].set_ylabel(metric, fontsize=18)
            axes[idx].tick_params(labelsize=18)

        # Hide any unused axes
        for idx in range(num_metrics, len(axes)):
            axes[idx].axis('off')

        # Annotating model performance metrics
        performance_str = "\n".join([f"{key}: {value[0]:.2f}" for key, value in self.performance.items()])
        fig.text(0.5, 0.01, performance_str, ha='center', va='center', fontsize=18,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="wheat"))

        # Set the main title for the entire plot
        fig.suptitle('Druggability and Performance Metrics', fontsize=25)

        # Adjust layout and display the plots
        plt.tight_layout(pad=4.0, rect=[0, 0.03, 1, 0.95])  # Adjust the rect to make space for suptitle
        plt.savefig(os.path.join(self.filepaths["plots"], 'metric_variations.png'), bbox_inches='tight')
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_idx", type=int, default=0)
    parser.add_argument("--root", type=str, default="./data/")
    parser.add_argument("--data", type=str, default="kiba")
    parser.add_argument("--protein_name", type=str, default="P21802")  # FGFR-2 with UniProt Id P21802 # Or pdb id
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--num_mols", type=int, default=1000)
    parser.add_argument("--chkpt_path", type=str, default="./checkpoints/")
    
    args = parser.parse_args()
    
    if args.device.isdigit():
        args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    elif args.device.lower() in ['cuda', 'cpu']:
        args.device = torch.device(args.device)
    else:
        raise ValueError(f"Unsupported device specified: {args.device}")

    return args

if __name__ == "__main__":
    args = parse_args()
    molecule_generator = MoleculeGenerator(args)
