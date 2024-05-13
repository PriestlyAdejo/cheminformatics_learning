import argparse
import os
import subprocess

def preprocess_protein_data(data_name, device):
    if data_name not in ["kiba", "davis"]:
        print("Invalid data_name. Use 'kiba' or 'davis'.")
        return

    # Navigate to the data folder
    data_folder = os.path.join("data", data_name)
    os.chdir(data_folder)

    # Extract protein 3D structures
    for file_name in ["prot_3d_for_Davis.tar.gz", "prot_3d_for_kiba.tar.gz"]:
        subprocess.run(["tar", "-xvzf", file_name])

    # Run data preprocessing
    subprocess.run(["python3", "process_protein_3d.py", "--data", data_name, "--device", device])

def train_binding_affinity(data_name, fold_idx):
    if data_name not in ["kiba", "davis"]:
        print("Invalid data_name. Use 'kiba' or 'davis'.")
        return
    if fold_idx not in [0, 1, 2, 3, 4]:
        print("Invalid fold_idx. Use values between 0 and 4.")
        return

    # Run binding affinity prediction training
    subprocess.run(["python3", "train_binding_affinity.py", "--data", data_name, "--fold_idx", str(fold_idx)])

def train_ligand_generation():
    # Assuming the pre-trained VAE checkpoint is in the same directory as train_vae.py
    if not os.path.isfile("vae_checkpoint.pth"):
        print("Pre-trained VAE checkpoint not found.")
        return

    # Run ligand generation training
    subprocess.run(["python3", "train_vae.py"])

def generate_ligands(num_mols):
    # Run ligand generation with the specified number of molecules
    subprocess.run(["python3", "generate_ligand.py", "--num_mols", str(num_mols)])

def generate_specific_target(target_name):
    # Assuming the PDB file is located in the /data directory
    pdb_file_path = os.path.join("data", f"{target_name}.pdb")

    if not os.path.isfile(pdb_file_path):
        print(f"PDB file for {target_name} not found.")
        return

    # Run ligand generation for the specific target
    subprocess.run(["python3", "generate_specific_target.py", "--protein_name", target_name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Target Processing and Ligand Generation")
    parser.add_argument("--step", type=str, choices=["preprocess", "train_affinity", "train_ligand", "generate_ligands", "generate_target"], required=True, help="Select the processing step.")
    parser.add_argument("--data_name", type=str, choices=["kiba", "davis"], default="kiba", help="Select the dataset (kiba or davis).")
    parser.add_argument("--fold_idx", type=int, choices=[0, 1, 2, 3, 4], default=0, help="Fold index for training (0-4).")
    parser.add_argument("--num_mols", type=int, default=100, help="Number of ligands to generate.")
    parser.add_argument("--target_name", type=str, default="", help="Name of the specific target for ligand generation.")
    args = parser.parse_args()

    if args.step == "preprocess":
        preprocess_protein_data(args.data_name, "cuda:0")
    elif args.step == "train_affinity":
        train_binding_affinity(args.data_name, args.fold_idx)
    elif args.step == "train_ligand":
        train_ligand_generation()
    elif args.step == "generate_ligands":
        generate_ligands(args.num_mols)
    elif args.step == "generate_target":
        generate_specific_target(args.target_name)
