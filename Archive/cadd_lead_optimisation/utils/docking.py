"""
Contains docking class.
"""

from pathlib import Path

import pandas as pd  # for creating dataframes and handling data

from .helpers import obabel, smina, autodock_gpu, binding_affinity_predictor, pdb, nglview, io


class Docking:
    """
    Automated docking process of the pipeline.
    Take in a Protein and a list of ligands, and
    dock each ligand on the protein, using the given specifications.

    Attributes
    ----------
    TODO
    """
    def __init__(
        self,
        protein_obj,
        list_ligand_obj,
        docking_specs_obj,
        docking_output_path,
        protein_pdb_code,
        frozen_data_filepath=None,
    ):
        """
        Initialize docking.

        Parameters
        ----------
        protein_obj : utils.Protein
            The protein to perform docking on.
        list_ligand_obj : list of utils.Ligand
            List of ligands to dock on the protein.
        docking_specs_obj : utils.Specs.Docking
            Specifications for the docking experiment.
        docking_output_path : str or pathlib.Path
            Output folder path to store the docking data in.
        frozen_data_filepath : str or pathlib.Path
            If existing data is to be used, provide the path to a folder
            containing the pdbqt files for
            (a) the protein `<PDBcode>_extracted_protein_ready_for_docking.pdbqt` and
            (b) all previously defined ligands/analogs `CID_<CID>.pdbqt`.
        """
        # Initial Path of the protein folder
        docking_output_path = Path(docking_output_path)
        protein_folder_path = docking_output_path / "protein" / f"PDB_{protein_obj.pdb_code}"
        io.create_folder(protein_folder_path)
        self.docking_program = docking_specs_obj.Docking.Program
        self.protein_pdb_code = protein_pdb_code

        self.pdb_filepath_extracted_protein = self.set_or_generate_pdb_file(
            protein_folder_path, protein_obj, frozen_data_filepath,
            suffix=f"{protein_obj.pdb_code}_extracted_protein.pdb"
        )

        self.pdbqt_filepath_extracted_protein = self.set_or_generate_pdbqt(
            protein_folder_path, protein_obj, frozen_data_filepath,
            suffix=f"{protein_obj.pdb_code}_extracted_protein_ready_for_docking.pdbqt",
            pdb_source=self.pdb_filepath_extracted_protein
        )
        
        temp_list_results_df = []
        temp_list_master_df = []

        for ligand in list_ligand_obj:
            ligand_folder_path = docking_output_path / "ligand" / f"CID_{ligand.cid}"
            io.create_folder(ligand_folder_path)

            ligand.pdbqt_filepath = self.set_or_generate_pdbqt(
                ligand_folder_path, ligand, frozen_data_filepath,
                suffix=f"CID_{ligand.cid}.pdbqt",
                smiles_source=ligand.remove_counterion()
            )

            ligand.docking_poses_filepath = ligand_folder_path / f"CID_{ligand.cid}_docking_poses.pdbqt"
            full_path_part = self.generate_full_path(ligand.docking_poses_filepath)

            # So model needs to generate all the data from raw log of smina or any other
            # docking program I decide to use.
            # this means I gotta use gpu version to generate things first for training
            # or find a dataset that has information I need already.
            # Then use it directly in the binding affinity pediction model.
            # Gets me some outputs to convert into dataframe
            # As usual this depends directly on the SPECS input dataframe and other
            # associated types.
            # No need to do docking for different binding sites on protein, assume best binding site
            # will be chosen automaticlly at later stage
            raw_log = self.perform_docking(ligand, protein_obj, docking_specs_obj, full_path_part)
            
            # Just splits ligand docking resutls for different poses.
            ligand.docking_poses_split_filepaths = obabel.split_multistructure_file(
                "pdbqt", ligand.docking_poses_filepath
            )

            self.process_docking_results(ligand, raw_log, temp_list_results_df, temp_list_master_df)

        self.results_dataframe = pd.concat(temp_list_results_df)
        self.master_df = pd.concat(temp_list_master_df)
        self.results_dataframe.to_csv(docking_output_path / f"DOCKING_RESULTS_SUMMARY_WITH_PDB_PROTEIN_{protein_obj.pdb_code}.csv")

    def set_or_generate_pdbqt(self, folder_path, obj, frozen_data_filepath, suffix, pdb_source=None, smiles_source=None):
        if frozen_data_filepath is not None:
            return folder_path / suffix
        else:
            pdbqt_filepath = folder_path / suffix
            if pdb_source:
                obabel.create_pdbqt_from_pdb_file(pdb_source, pdbqt_filepath)
            elif smiles_source:
                obabel.create_pdbqt_from_smiles(smiles_source, pdbqt_filepath)
            return pdbqt_filepath

    def set_or_generate_pdb_file(self, folder_path, obj, frozen_data_filepath, suffix):
        pdb_filepath = folder_path / suffix
        if not frozen_data_filepath:
            pdb.extract_molecule_from_pdb_file(
                "protein", obj.pdb_filepath, pdb_filepath
            )
        return pdb_filepath

    def generate_full_path(self, filepath):
        path_parts = filepath.parts
        return "/".join(path_parts[:-1]) + "/" + path_parts[-1].split('.')[0]

    def perform_docking(self, ligands, protein_obj, docking_specs_obj, full_path_part):
        if self.docking_program == "smina":
            result = smina.dock(
                ligands.pdbqt_filepath,
                self.pdbqt_filepath_extracted_protein,
                protein_obj.binding_site_coordinates["center"],
                protein_obj.binding_site_coordinates["size"],
                full_path_part,
                output_format="pdbqt",
                num_poses=docking_specs_obj.num_poses_per_ligand,
                exhaustiveness=docking_specs_obj.exhaustiveness,
                random_seed=docking_specs_obj.random_seed,
                log=True,
            )
        elif self.docking_program == "binding_affinity_predictor":
            result = binding_affinity_predictor.dock(
                [li_obj.smiles for li_obj in ligands],
                self.protein_pdb_code,
                fold_idx=0,
                dataset="kiba",
                random_seed=docking_specs_obj.random_seed,
                log=True,
            )
        return result

    def process_docking_results(self, ligand, raw_log, temp_list_results_df, temp_list_master_df):
        if self.docking_program == "smina":
            df = smina.convert_log_to_dataframe(raw_log)
        elif self.docking_program == "binding_affinity_predictor":
            df = binding_affinity_predictor.convert_log_to_dataframe(raw_log)
        
        ligand.dataframe_docking = df.copy()
        self.assign_ligand_properties(ligand, df)
        df["CID"] = ligand.cid
        df["drug_score_total"] = ligand.drug_score_total
        df.set_index(["CID", df.index], inplace=True)
        master_df = df.copy()
        master_df["filepath"] = ligand.docking_poses_split_filepaths
        temp_list_results_df.append(df)
        temp_list_master_df.append(master_df)

    def assign_ligand_properties(self, ligand, df):
        ligand.binding_affinity_best = df["affinity[kcal/mol]"].min()
        ligand.binding_affinity_mean = df["affinity[kcal/mol]"].mean()
        ligand.binding_affinity_std = df["affinity[kcal/mol]"].std()
        ligand.docking_poses_dist_rmsd_lb_mean = df["dist from best mode_rmsd_l.b"].mean()
        ligand.docking_poses_dist_rmsd_lb_std = df["dist from best mode_rmsd_l.b"].std()
        ligand.docking_poses_dist_rmsd_ub_mean = df["dist from best mode_rmsd_u.b"].mean()
        ligand.docking_poses_dist_rmsd_ub_std = df["dist from best mode_rmsd_u.b"].std()
        for attr_name in ["binding_affinity_best", "binding_affinity_mean", "binding_affinity_std", 
                        "docking_poses_dist_rmsd_lb_mean", "docking_poses_dist_rmsd_lb_std", 
                        "docking_poses_dist_rmsd_ub_mean", "docking_poses_dist_rmsd_ub_std"]:
            ligand.dataframe.loc[attr_name] = getattr(ligand, attr_name)

    def visualize_all_poses(self):
        """
        Visualize docking poses of a all analogs, using NGLView.

        Returns
        -------
        nglview.widget.NGLWidget
            Interactive viewer of all analogs' docking poses,
            sorted by their binding affinities.
        """
        df = self.master_df.sort_values(by=["affinity[kcal/mol]", "CID", "mode"])
        self.visualize(df)
        return

    def visualize_analog_poses(self, cid):
        """
        Visualize docking poses of a certain analog, using NGLView.

        Parameters
        ----------
        cid : str or int
            CID of the analog.

        Returns
        -------
        nglview.widget.NGLWidget
            Interactive viewer of given analog's docking poses,
            sorted by their binding affinities.
        """
        df = self.master_df.xs(str(cid), level=0, axis=0, drop_level=False)
        self.visualize(df)
        return

    def visualize(self, fitted_master_df):
        """
        Visualize any collection of docking poses, using NGLView.

        Parameters
        ----------
        fitted_master_df : pandas.DataFrame
            Any section of the master docking dataframe,
            stored under self.master_df.

        Returns
        -------
        nglview.widget.NGLWidget
            Interactive viewer of given analog's docking poses,
            sorted by their binding affinities.
        """
        list_docking_poses_labels = list(
            map(lambda x: f"{x[0]} - {x[1]}", fitted_master_df.index.tolist())
        )
        nglview.docking(
            self.pdb_filepath_extracted_protein,
            fitted_master_df["filepath"].tolist(),
            list_docking_poses_labels,
            fitted_master_df["affinity[kcal/mol]"].tolist(),
        )
        return
