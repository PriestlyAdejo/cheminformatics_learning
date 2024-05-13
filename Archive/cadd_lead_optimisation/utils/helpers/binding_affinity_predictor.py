"""
Set of functions for communicating with the Smina docking program,
and extracting data from its log file.
"""

import subprocess  # for creating shell processes (needed to communicate with Smina program)
import pandas as pd  # for creating dataframes and handling data
from predict_specific_binding_affinity import SpecificBindingAffinityPredictor


def dock(
    ligand_smiles,
    protein_pdb_code,
    output_path,
    dataset="kiba",
    fold_idx=0,
    device="cuda",
    random_seed=None,
    log=True,
):
    binding_affinity_predictor_command = (
        [
            "python3",
            "predict_specific_binding_affinity.py",
            "--data", dataset,
            "--fold_idx", str(fold_idx),
            "--device", device,
            "--ligand_smiles", smiles_arg,
            "--protein_id", protein_pdb_code,
        ]
    )

    output_text = subprocess.check_output(
        binding_affinity_predictor_command,
        universal_newlines=True,
    )
    return output_text


def convert_log_to_dataframe(raw_log):
    """
    Convert docking's raw output log into a pandas.DataFrame, with specific adjustments.

    Parameters
    ----------
    raw_log : str
        Raw output log generated after docking.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing columns 'mode', 'affinity[kcal/mol]',
        'dist from best mode_rmsd_l.b', and 'dist from best mode_rmsd_u.b'
        for each generated docking pose.
    """

    # Extract the predictions part of the log
    predictions_str = raw_log.split("############################ Results ############################")[1].split("LEN INPUTS:")[0].split("PREDICTIONS:")[-1].strip()

    # Convert the string representation of the list into a Python list
    predictions = eval(predictions_str)

    # Construct the DataFrame
    df = pd.DataFrame(predictions, columns=['affinity[kcal/mol]'])
    
    # Adjust the DataFrame according to the requirements
    df['affinity[kcal/mol]'] = df['affinity[kcal/mol]'].apply(lambda x: 0 if x > 0 else x)
    df['mode'] = 1
    df['dist from best mode_rmsd_l.b'] = 1
    df['dist from best mode_rmsd_u.b'] = 1

    # Reorder the columns as specified
    df = df[['mode', 'affinity[kcal/mol]', 'dist from best mode_rmsd_l.b', 'dist from best mode_rmsd_u.b']]

    df.index = df["mode"]
    df.drop("mode", axis=1, inplace=True)
    
    return df
