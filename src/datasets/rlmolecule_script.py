import pandas as pd

from rdkit import Chem

from rdkit.Chem import QED, Crippen

from rdkit.Contrib.SA_Score import sascorer
from rdkit.Contrib.NP_Score import npscorer

from joblib import Parallel, delayed



# Function to calculate QED and logP for a single SMILES string

def calculate_properties(smile):

    try:

        mol = Chem.MolFromSmiles(smile)

        if mol:  # Check if the molecule is valid

            # mol = Chem.AddHs(mol)
            # AllChem.EmbedMolecule(mol)
            # if AllChem.EmbedMolecule(mol, randomSeed=self.seed) !=0:
            #     return None, None, None
            # AllChem.UFFOptimizeMolecule(mol)
            # mol = Chem.RemoveHs(mol)

            # x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long).view(-1, 1)
            # pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)

            # if x.shape[0] != mol.GetNumAtoms():
            #     return None, None, None

            qed = QED.qed(mol)

            logp = Crippen.MolLogP(mol)

            sascore = sascorer.calculateScore(mol)

            return qed, logp, sascore

    except Exception as e:

        print(f"Error processing SMILES {smile}: {e}")

    return None, None, None  # Return None for invalid or problematic SMILES



# Function to compute QED and logP values in parallel

def compute_properties_for_all(input_csv, output_csv):

    try:

        # Load the CSV into a DataFrame

        df = pd.read_csv(input_csv)

        df.drop_duplicates(inplace=True)

        df.sort_values(by='score_Docking', ascending=False, inplace=True)



        # Check the size of the DataFrame

        print(f"The input DataFrame contains {df.shape[0]} rows and {df.shape[1]} columns.")



        # Ensure the required column is present

        if 'smiles' not in df.columns:

            raise ValueError("The input CSV must contain a 'smiles' column.")



        # Use joblib's Parallel and delayed to parallelize property calculation

        properties = Parallel(n_jobs=-1)(delayed(calculate_properties)(smile) for smile in df['smiles'])



        # Split properties into separate columns for QED and logP

        qed_values, logp_values, sascore_values = zip(*properties)



        # Assign the computed values to the DataFrame

        df['QED'] = qed_values

        df['LogP'] = logp_values

        df['SA'] = sascore_values


        df = df.dropna(subset=['QED', 'LogP', 'SA'])
        # Save the updated DataFrame to a new CSV

        df.to_csv(output_csv, index=False)



        print(f"The QED and LogP values for all SMILES have been computed and saved to {output_csv}")

    except Exception as e:

        print(f"An error occurred: {e}")



# Example usage

input_csv_path = "compounds.csv"

output_csv_path = "output.csv"

compute_properties_for_all(input_csv_path, output_csv_path)