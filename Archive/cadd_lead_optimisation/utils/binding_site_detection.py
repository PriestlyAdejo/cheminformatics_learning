"""
Contains binding site detection class.
"""

from enum import Enum  # for creating enumeration classes
from pathlib import Path

from .helpers import dogsitescorer, pdb, nglview, io


class BindingSiteDetection:
    """
    Automated binding site detection process of the pipeline.
    Take in the Protein object, Specs.BindingSite object and
    the corresponding output path, and automatically run all the necessary calculations
    to output the suitable binding site coordinates based on the input specifications of the
    project.

    Attributes
    ----------
    TODO see `Consts` class. More attributes?
    """

    class Consts:
        class DefinitionMethods(Enum):
            DETECTION = "detection"
            LIGAND = "ligand"
            COORDINATES = "coordinates"

    def __init__(self, protein_obj, binding_site_specs_obj, binding_site_output_path):
        """
        Initialize binding site detection.

        Parameters
        ----------
        protein_obj : utils.Protein
            The Protein object of the project.
        binding_site_specs_obj : utils.Specs.BindingSite
            The binding site specification data-class of the project.
        binding_site_output_path : str or pathlib.Path
            Output path of the project's binding site information.
        """

        self.output_path = Path(binding_site_output_path)
        self.Protein = protein_obj
        # derive the relevant function name from definition method
        definition_method_name = f"compute_by_{binding_site_specs_obj.definition_method.name.lower()}"
        # get the function from its name
        definition_method = getattr(self, definition_method_name)
        # call the function
        definition_method(protein_obj, binding_site_specs_obj, binding_site_output_path)

    def compute_by_coordinates(self, protein_obj, binding_site_specs_obj, binding_site_output_path=None):
        """
        Extract the binding site coordinates specified in input specifications. 
        The function is called when binding site definition method is `coordinates`.
        
        Parameters
        ----------
        protein_obj : utils.Protein
            The Protein object of the project.
        binding_site_specs_obj : utils.Specs.BindingSite
            The binding site specification data-class of the project.
        
        Returns
        -------
        None
            The coordinates are assigned to protein_obj.binding_site_coordinates.
        """
        protein_obj.binding_site_coordinates = binding_site_specs_obj.coordinates

    def compute_by_ligand(self, protein_obj, binding_site_specs_obj, binding_site_output_path):
        """
        Calculate the coordinates of the binding site from the position
        of the co-crystallized ligand in the input protein structure.
        The function is called when binding site definition method is 'ligand'.
        
        Parameters
        ----------
        protein_obj : utils.Protein
            The Protein object of the project.
        binding_site_specs_obj : utils.Specs.BindingSite
            The binding site specification data-class of the project.
        binding_site_output_path : str or pathlib.Path
            Output path of the project's binding site information.
        
        Returns
        -------
        None
            The coordinates are assigned to protein_obj.binding_site_coordinates.
        """
        ligand_object = pdb.extract_molecule_from_pdb_file(
            binding_site_specs_obj.protein_ligand_id,
            protein_obj.pdb_filepath,
            binding_site_output_path / binding_site_specs_obj.protein_ligand_id,
        )
        
        # calculate the geometric center of the molecule,
        # which represents the center of the rectangular box,
        # as well as the length of the molecule in each dimension,
        # which corresponds to the edge lengths of the rectangular box.
        # Also, we will add a 5 Å buffer in each dimension to allow the
        # correct placements of ligands that are bigger than the co-crystallized ligands
        # or bind in a different fashion.
        pocket_center = (
            ligand_object.positions.max(axis=0) + ligand_object.positions.min(axis=0)
        ) / 2
        pocket_size = ligand_object.positions.max(axis=0) - ligand_object.positions.min(axis=0) + 5
        pocket_coordinates = {
            "center": pocket_center.tolist(),
            "size": [pocket_size.tolist()],
        }
        protein_obj.binding_site_coordinates = pocket_coordinates
        return

    def compute_by_detection(self, protein_obj, binding_site_specs_obj, binding_site_output_path):
        """
        Derive the name of the function corresponding to the user-specified detection method,
        and call that function, passing the necessary arguments, to perform a detection job.
        The function is called when binding site definition method is 'detection'.
        
        Parameters
        ----------
        protein_obj : utils.Protein
            The Protein object of the project.
        binding_site_specs_obj : utils.Specs.BindingSite
            The binding site specification data-class of the project.
        binding_site_output_path : str or pathlib.Path
            Output path of the project's binding site information.
        
        Returns
        -------
        None
            Another function is called.
        """
        # derive the relevant function name from detection method
        detection_method_name = f"detect_by_{binding_site_specs_obj.detection_method.name.lower()}"
        # get the function from its name
        detection_method = getattr(self, detection_method_name)
        # call the function
        detection_method(protein_obj, binding_site_specs_obj, binding_site_output_path)

    def detect_by_dogsitescorer(self, protein_obj, binding_site_specs_obj, binding_site_output_path):
        """
        Calculate the coordinates of the binding site by submitting a
        detection job to DoGSiteScorer, and analyzing the results to find
        the best binding site according to input specifications.
        The function is called when binding site definition method is 'detection',
        and detection method is 'dogsitescorer'.
        
        Parameters
        ----------
        protein_obj : utils.Protein
            The Protein object of the project.
        binding_site_specs_obj : utils.Specs.BindingSite
            The binding site specification data-class of the project.
        binding_site_output_path : str or pathlib.Path
            Output path of the project's binding site information.
        
        Returns
        -------
        None
            The coordinates are assigned to protein_obj.binding_site_coordinates.
        """
        if hasattr(protein_obj, "pdb_code"):
            self.dogsitescorer_pdb_id = protein_obj.pdb_code
        elif hasattr(protein_obj, "pdb_filepath"):
            self.dogsitescorer_pdb_id = dogsitescorer.upload_pdb_file(protein_obj.pdb_filepath)
        else:
            raise AttributeError(f"Protein has neither `pdb_code` nor `pdb_filepath attributes.")

        # try to get the chain_id for binding site detection if it's available in input data
        if binding_site_specs_obj.protein_chain_id != "":
            if hasattr(protein_obj, "chains") and binding_site_specs_obj.protein_chain_id in protein_obj.chains:
                self.dogsitescorer_chain_id = binding_site_specs_obj.protein_chain_id
            else:
                raise ValueError(
                    f"The input protein chain-ID ({binding_site_specs_obj.protein_chain_id}) "
                    f"does not exist in the input protein. Existing chains are: {protein_obj.chains}"
                )
        # if chain_id is not in input data, but the protein has chain-IDs
        # set it to the first chain-ID found in the list of chain-IDs
        elif hasattr(protein_obj, "chains") and len(protein_obj.chains)>0:
            self.dogsitescorer_chain_id = protein_obj.chains[0]
        # if no chain is found in PDB file either, leave the chain-ID empty
        else:
            self.dogsitescorer_chain_id = ""

        # create a list of ligand ids in the DoGSiteScorer format
        list_dogsitescorer_ligand_ids = []
        list_ligands_heavy_atom_count = []
        for ligand in protein_obj.ligands:
            list_ligands_heavy_atom_count.append(ligand[-1])
            if len(ligand) == 3:
                list_dogsitescorer_ligand_ids.append(
                    ligand[0] + "_" + ligand[1][0] + "_" + ligand[1][1:]
                )
            else:
                list_dogsitescorer_ligand_ids.append(ligand[0] + "_" + ligand[1] + "_" + ligand[2])

        # try to get the ligand_id for detection calculation if it's available in input data
        if binding_site_specs_obj.protein_ligand_id != "":
            ligand_id = binding_site_specs_obj.protein_ligand_id
            # check if the given ligand_id is already given in the DoGSiteScorer format
            if "_" in ligand_id:
                if ligand_id in list_dogsitescorer_ligand_ids:
                    self.dogsitescorer_ligand_id = ligand_id
                    self.dogsitescorer_chain_id = ligand_id.split("_")[1][0]
                else:
                    raise ValueError(
                        f"The input ligand-ID ({binding_site_specs_obj.protein_ligand_id}) "
                        f"does not exist in the input protein. Existing ligand-IDs are: "
                        f"{[ligand[0] for ligand in protein_obj.ligands]}"
                    )
            else:
                self.dogsitescorer_ligand_id = None
                for ligand in list_dogsitescorer_ligand_ids:
                    if ligand.split("_")[0] == ligand_id and (
                        ligand.split("_")[1][0] == self.dogsitescorer_chain_id
                    ):
                        self.dogsitescorer_ligand_id = ligand
                        break
                    else:
                        pass

                if self.dogsitescorer_ligand_id is None:
                    raise ValueError(
                        f"The input ligand-ID ({binding_site_specs_obj.protein_ligand_id}) "
                        f"does not exist in the input protein. Existing ligand-IDs are: "
                        f"{[ligand[0] for ligand in protein_obj.ligands]}"
                    )

        else:
            index_of_heaviest_ligand = list_ligands_heavy_atom_count.index(
                max(list_ligands_heavy_atom_count)
            )
            self.dogsitescorer_ligand_id = list_dogsitescorer_ligand_ids[index_of_heaviest_ligand]

        self.dogsitescorer_binding_sites_df = dogsitescorer.submit_job(
            self.dogsitescorer_pdb_id,
            self.dogsitescorer_ligand_id,
            self.dogsitescorer_chain_id,
        )
        
        dogsitescorer.save_binding_sites_to_file(
            self.dogsitescorer_binding_sites_df, binding_site_output_path
        )
        
        self.dogsitescorer_binding_sites_df = self.filter_invalid_binding_sites(
            self.dogsitescorer_binding_sites_df,
            binding_site_output_path
        )
        
        self.best_binding_site_name = dogsitescorer.select_best_pocket(
            self.dogsitescorer_binding_sites_df,
            binding_site_specs_obj.selection_method.value,
            binding_site_specs_obj.selection_criteria,
        )

        self.best_binding_site_data = self.dogsitescorer_binding_sites_df.loc[
            self.best_binding_site_name
        ]

        target_output_path = self.get_binding_site_name_output_path(binding_site_output_path, self.best_binding_site_name)

        # Calculate the best binding site coordinates
        self.best_binding_site_coordinates = dogsitescorer.calculate_pocket_coordinates_from_pocket_pdb_file(
            target_output_path / self.best_binding_site_name  # Assuming the file format is .pdb
        )

        protein_obj.binding_site_coordinates = self.best_binding_site_coordinates
        return

    def filter_invalid_binding_sites(self, df, binding_site_output_path):
        # List to store binding site names with valid coordinates
        valid_binding_site_names = []

        # Iterate over all binding site names and check their coordinates
        for name, _ in df.iterrows():
            # Split the binding site name to determine the pocket and subpocket
            target_output_path = self.get_binding_site_name_output_path(binding_site_output_path, name)

            coordinates = dogsitescorer.calculate_pocket_coordinates_from_pocket_pdb_file(
                target_output_path / name 
            )
            # Check if all coordinates have 3 dimensions
            if all(len(v) == 3 for v in coordinates.values()):
                valid_binding_site_names.append(name)

        # Filter the binding_sites_df to keep only those with valid coordinates
        df = df[df.index.isin(valid_binding_site_names)]

        # Check if the DataFrame is empty after filtering
        if df.empty:
            raise ValueError("All binding sites were filtered out due to invalid coordinates.")

        return df
    
    def get_binding_site_name_output_path(self, binding_site_output_path, binding_site_name):
        # Split the best binding site name to determine the pocket and subpocket
        parts = binding_site_name.split("_")
        pocket = f"{parts[0]}_{parts[1]}"
        subpocket = "_".join(parts[2:]) if len(parts) > 2 else None

        # Create a unique folder for the best binding site pocket and subpocket
        pocket_output_path = Path(binding_site_output_path) / "binding_site" / f"pocket_{pocket}"
        io.create_folder(pocket_output_path)
        if subpocket:
            subpocket_output_path = pocket_output_path / f"subpocket_{subpocket}"
            io.create_folder(subpocket_output_path)
            target_output_path = subpocket_output_path
        else:
            target_output_path = pocket_output_path
        
        return target_output_path

    def visualize(self, pocket_name):
        """
        Visualize a detected pocket.

        Parameters
        ----------
        pocket_name : str
            Name of the detected pocket, which has a corresponding CCP4 file
            with the same name stored in the output folder for binding site data.

        Returns
        -------
        nglview.widget.NGLWidget
            Viewer showing the given pocket.
        """

        # Determine the input type and value for the protein
        protein_input_type, protein_input_value = (
            ("pdb_code", self.Protein.pdb_code) if hasattr(self.Protein, "pdb_code") 
            else ("pdb_filepath", self.Protein.pdb_filepath)
        )

        # Get the output path for the binding site
        binding_site_output_path = self.get_binding_site_name_output_path(self.output_path, pocket_name)

        # Construct the path to the CCP4 and PNG file
        ccp4_file_path = binding_site_output_path / f"{pocket_name}"
        png_file_path = binding_site_output_path / f"{pocket_name}.png"

        # Create the viewer
        viewer = nglview.binding_site(
            protein_input_type,
            protein_input_value,
            ccp4_file_path,
            png_file_path
        )
        return viewer

    def visualize_best(self):
        """
        Visualize the selected binding pocket.
        The binding pocket should have a corresponding CCP4 file
        with the same name stored in the output folder for binding site data.

        Returns
        -------
        nglview.widget.NGLWidget
            Viewer showing the given pocket.
        """
        return self.visualize(self.best_binding_site_name)
    
    def visualize_all(self):
        """
        Visualize all the binding pockets from input df.
        The binding pocket should have a corresponding CCP4 file
        with the same name stored in the output folder for binding site data.

        Returns
        -------
        nglview.widget.NGLWidget
            Viewer showing the given pocket.
        """
        # Iterate over all binding site names and check their coordinates
        viewers = []
        for name, _ in self.dogsitescorer_binding_sites_df.iterrows():
            view = self.visualize(name)
            viewers.append(view)
        return viewers
