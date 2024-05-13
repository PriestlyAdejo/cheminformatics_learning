import os
import argparse
import atom3d.util.formats as fo
from tqdm import tqdm
import selfies as sf
import pandas as pd
import torch
from utils.protein_utils import featurize_as_graph
from utils.transform import BaseTransform
import shutil

def remove_directory_contents(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            remove_directory_contents(item_path)
            os.rmdir(item_path)
        else:
            os.remove(item_path)

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default="./data/")
parser.add_argument("--data", type=str, default="kiba")
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()
transform = BaseTransform(device=args.device)

roots = ["data/davis/prot_3d_for_davis", "data/kiba/prot_3d_for_kiba"]

if args.data == "davis":
    root = "data/davis/prot_3d_for_davis"
else:
    root = "data/kiba/prot_3d_for_kiba"

# Ensure the res_graph directory exists and is empty
for iroot in roots:
    res_graph_dir = os.path.join(iroot, "res_graph")
    if os.path.exists(res_graph_dir):
        remove_directory_contents(res_graph_dir)
        os.rmdir(res_graph_dir)
        print(f"Removed: {res_graph_dir}")
    os.makedirs(res_graph_dir)

# Process each folder
folders = os.listdir(root)
for folder in tqdm(folders):
    path = os.path.join(root, folder)
    if os.path.isfile(path) and path.endswith('.pdb'):  # Ensure it's a PDB file
        protein_df = fo.bp_to_df(fo.read_pdb(path))
        protein_graph = featurize_as_graph(protein_df)

        # Save the file
        torch.save(protein_graph, os.path.join(res_graph_dir, f"{folder}.pt"))

# Move the res_graph directory up one level after processing all folders
parent_dir = os.path.dirname(root)
new_location = os.path.join(parent_dir, "res_graph")

if os.path.exists(new_location):
    remove_directory_contents(new_location)
    os.rmdir(new_location)

shutil.move(res_graph_dir, new_location)