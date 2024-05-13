"""
Contains the Ligand class of the pipeline.
"""

from enum import Enum  # for creating enumeration classes
from pathlib import Path

import pandas as pd  # for creating dataframes and handling data
from rdkit.Chem import PandasTools # for displaying structures inside Pandas DataFrame
PandasTools.RenderImagesInAllDataFrames(True)

from .helpers import pubchem, rdkit, io
from .helpers.rdkit import calculate_druglikeness


class Ligand:
    """
    Ligand object with properties as attributes and methods to visualize and work with ligands.
    Take a ligand identifier type and corresponding value,
    and create a Ligand object, while assigning some properties as attributes.

    Attributes
    ----------
    dataframe : Pandas DataFrame
        Dataframe containing the most important data available for the ligand.
        Each time a process is performed on the ligand (e.g. docking), 
        the important results are added to the dataframe.
    rdkit_obj : rdkit.Chem.rdchem.Mol
        RDKit molecule object for the ligand with its own set of attributes.
    
    TODO see `Consts` class. More attributes?
    """

    class Consts:
        """
        Available properties that are assigned as instance attributes upon instantiation.
        """

        class IdentifierTypes(Enum):
            NAME = "name"
            IUPAC_NAME = "iupac_name"
            SMILES = "smiles"
            CID = "cid"
            INCHI = "inchi"
            INCHIKEY = "inchikey"

    def __init__(self, identifier_type, identifier_value, ligand_output_path):
        """
        Parameters
        ----------
        identifier_type : enum 'InputTypes' from the 'Consts.Ligand' class
            Type of the ligand identifier, e.g. InputTypes.SMILES.
        indentifier_value : str
            Value of the ligand identifier, e.g. its SMILES.
        ligand_output_path : str or pathlib.Path
            Output path of the project for ligand data.
        """

        self.dataframe = pd.DataFrame(columns=["Value"])
        self.dataframe.index.name = "Property"
        if identifier_value:
            setattr(self, identifier_type.name.lower(), identifier_value)

        # setattr(self, identifier_type.name.lower(), identifier_value)
        for identifier in self.Consts.IdentifierTypes:
            new_id = pubchem.convert_compound_identifier(
                identifier_type.value, identifier_value, identifier.value
            )
            if len(new_id) > 0:
                setattr(self, identifier.value, new_id)
            self.dataframe.loc[identifier.value] = new_id
            
        self.rdkit_obj = rdkit.create_molecule_object("smiles", self.smiles)
        dict_of_properties = calculate_druglikeness(self.rdkit_obj)
        for property_ in dict_of_properties:
            setattr(self, property_, dict_of_properties[property_])
            self.dataframe.loc[property_] = dict_of_properties[property_]
        
        # Saving things to folders etc
        ligand_output_path = Path(io.create_folder(ligand_output_path / "ligand" / f"CID_{self.cid}")) 
        self.save_as_image(ligand_output_path / f"IMAGE_CID_{self.cid}")
        self.dataframe.to_csv(ligand_output_path / f"ORIGINAL_DF_CID_{self.cid}.csv")
        self.dataframe = self()
        self.dataframe.to_csv(ligand_output_path / f"CONCAT_DF_CID_{self.cid}.csv")
        
    def __repr__(self):
        return f"<Ligand CID: {self.cid}>"

    def __call__(self):
        # Exclude certain attributes (like 'dataframe') when building attr_dict
        exclude_attrs = {'dataframe'}
        attr_dict = {attr: getattr(self, attr) for attr in dir(self) if not attr.startswith('__') and not callable(getattr(self, attr)) and attr not in exclude_attrs}
        attr_df = pd.DataFrame.from_dict(attr_dict, orient='index', columns=['Value']).reset_index().rename(columns={'index': 'Property'})

        # Concatenate with the existing dataframe
        concat_df = pd.concat([self.dataframe.reset_index(), attr_df]).set_index('Property')

        # Process to remove duplicate columns, keeping non-empty values where possible
        unique_properties = {}
        for property_name, value in reversed(list(concat_df.itertuples(index=True))):
            if property_name not in unique_properties:
                unique_properties[property_name] = value
            elif pd.isna(unique_properties[property_name]) and not pd.isna(value):
                unique_properties[property_name] = value

        # Create new DataFrame from unique_properties
        unique_df = pd.DataFrame(list(unique_properties.items()), columns=['Property', 'Value']).set_index('Property')

        return unique_df

    def remove_counterion(self):
        """
        Remove the counter-ion from the SMILES of a salt.

        Returns
        -------
        str
            SMILES of the molecule without its counter-ion.
        """
        
        # SMILES of salts contain a dot, separating the anion and the cation
        if "." in self.smiles:  
            ions = self.smiles.split(".")
            length_ions = list(map(len, ions))
            # The parent molecule is almost always larger than its corresponding counter-ion
            molecule_index = length_ions.index(max(length_ions))  
            molecule_smiles = ions[molecule_index]
        else:
            molecule_smiles = self.smiles
        return molecule_smiles

    def dice_similarity(self, mol_obj):
        """
        Calculate Dice similarity between the ligand and another input ligand,
        based on 4096-bit Morgan fingerprints with a radius of 2.

        Parameters
        ----------
        mol_obj : RDKit molecule object
            The molecule to calculate the Dice similarity with.

        Returns
        -------
        float
            Dice similarity between the two ligands.
        """
        return rdkit.calculate_similarity_dice(self.rdkit_obj, mol_obj)

    def save_as_image(self, filepath):
        """
        Save the ligand as image.

        Parameters
        ----------
        filepath : str or pathlib.Path object
            Full filepath of the image to be saved.

        Returns
        -------
        None
        """
        rdkit.save_molecule_image_to_file(self.rdkit_obj, filepath)

    def save_3D_structure_as_SDF_file(self, filepath):
        """
        Generate a 3D conformer and save as SDF file.

        Parameters
        ----------
        filepath : str or pathlib.Path object
            Full filpath to save the image in.

        Returns
        -------
        None
        """
        rdkit.save_3D_molecule_to_SDfile(self.rdkit_obj, filepath)
