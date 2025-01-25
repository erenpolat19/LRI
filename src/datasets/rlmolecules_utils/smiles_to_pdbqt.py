from rdkit import Chem

from rdkit.Chem import AllChem

import os

import subprocess

from multiprocessing import Pool, cpu_count

import pandas as pd

from pathlib import Path

import tempfile



# Set up base directories




def process_smiles_to_pdb(smiles_data):

    mol, cnt, logfile = smiles_data

    try:

        # Create ligand object from SMILES string

        lig = Chem.MolFromSmiles(mol)

        if lig is None:

            raise ValueError(f"Invalid SMILES string: {mol}")



        # Protonate and embed ligand

        protonated_lig = Chem.AddHs(lig)

        embedding_status = AllChem.EmbedMolecule(protonated_lig)

        if embedding_status != 0:

            raise ValueError("Embedding failed")



        # Optimize ligand geometry

        #AllChem.UFFOptimizeMolecule(protonated_lig)
        AllChem.MMFFOptimizeMolecule(protonated_lig, maxIters=1000)


        # Save ligand as PDB

        pdbfile = LIGANDS_PDB_DIR / f"drug_{cnt}.pdb"

        Chem.MolToPDBFile(protonated_lig, str(pdbfile))



    except Exception as e:

        error_message = f"Error processing molecule {cnt} (SMILES: {mol}): {e}\n"

        with open(logfile, "a") as f:

            f.write(error_message)



def convert_pdb_to_pdbqt(pdb_data):

    pdbfile, cnt, logfile = pdb_data

    original_dir = os.getcwd()  # Save the current directory

    try:

        os.chdir(LIGANDS_PDB_DIR)

        pdbqtfile = LIGANDS_PDBQT_DIR / f"drug_{cnt}.pdbqt"



        cmd = [

            "/home/eshagupta/bin/mgltools_x86_64Linux2_1.5.7/bin/pythonsh",

            "/home/eshagupta/bin/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",

            "-l", str(pdbfile.resolve()), "-o", str(pdbqtfile.resolve())

        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:

            raise RuntimeError(f"Error converting {pdbfile} to PDBQT:\n{result.stderr}")

    except Exception as e:

        error_message = f"Error converting molecule {cnt} (PDB: {pdbfile}): {e}\n"

        with open(logfile, "a") as f:

            f.write(error_message)

        

    finally:

        # Restore the original directory

        os.chdir(original_dir)



def process_smiles_file(filename, output_dir, error_dir, ncpu=None):

    logfile = os.path.join(error_dir, "error.log")

    os.makedirs(error_dir, exist_ok=True)



    # Load SMILES strings

    smiles_df = pd.read_csv(filename)

    # Sort the DataFrame by the 'score_Docking' column in ascending order
    smiles_df = smiles_df.sort_values(by='score_Docking')

    # Select the top 4000 and bottom 4000 rows
    smiles_df = smiles_df.tail(5000)

    # Reset the index of the resulting DataFrame
    smiles_df = smiles_df.reset_index(drop=True)

    smiles_strs = smiles_df['smiles'].values

    # Prepare data for multiprocessing

    ncpu = ncpu or cpu_count()

    smiles_data = [(mol, idx, logfile) for idx, mol in zip(smiles_df.index, smiles_strs)]



    # Process in parallel to generate PDB files

    with Pool(ncpu) as pool:

        pool.map(process_smiles_to_pdb, smiles_data)

        pool.close()  # Ensure all PDB files are generated

        pool.join()

    

    # Stage 2: Convert PDB files to PDBQT files

    pdb_files = [

        (pdbfile, pdbfile.stem.split('_')[1], logfile)  # Extract `drug_x` number from filename

        for pdbfile in LIGANDS_PDB_DIR.glob("drug_*.pdb")  # Match all `drug_*.pdb` files

    ]



    # with Pool(ncpu) as pool:

    #     pool.map(convert_pdb_to_pdbqt, pdb_files)

    #     pool.close()  # Ensure all PDB files are generated

    #     pool.join()



if __name__ == '__main__':

    import argparse



    parser = argparse.ArgumentParser(description='Convert SMILES strings to PDBQT files with multiprocessing.')

    parser.add_argument('-i', '--input', metavar='FILENAME', type=str, required=True,

                        help='Input CSV file containing SMILES strings.')

    parser.add_argument('-e', '--error_dir', metavar='DIRNAME', type=str, required=True,

                        help='Output directory to save error log.')

    parser.add_argument('-o', '--output_dir', metavar='ERRORDIRNAME', type=str, required=True,

                        help='Output prefix to save pdb log.')

    parser.add_argument('-c', '--ncpu', type=int, default=None,

                        help='Number of CPUs to use for multiprocessing (default: all available CPUs).')



    args = parser.parse_args()

    
    BASE_DIR = Path().resolve()

    LIGANDS_PDB_DIR = BASE_DIR / (args.output_dir + "_pdb")

    LIGANDS_PDBQT_DIR = BASE_DIR / (args.output_dir + "_pdbqt")



    LIGANDS_PDB_DIR.mkdir(exist_ok=True)

    #LIGANDS_PDBQT_DIR.mkdir(exist_ok=True)

    if not os.path.exists(args.input):

        raise FileNotFoundError(f"Input file {args.input} does not exist.")

    process_smiles_file(args.input, args.output_dir, args.error_dir, args.ncpu)