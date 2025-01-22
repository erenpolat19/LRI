import sys
sys.path.append('../')

import os
import yaml
import shutil
import os.path as osp
from tqdm import tqdm
import py3Dmol

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset, Data

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import random

class RLMolecules(InMemoryDataset):
    def __init__(self, root, seed, data_config = None):
        self.seed = seed
        self.COUNT = 1000
        self.data_config = data_config
        self.ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'Na', 'Ca', 'I', 'B', 'H', '*']
        self.data_list = None
        super().__init__(root)
        self.data, self.slices, self.idx_split = torch.load(self.processed_paths[0])
        self.x_dim = self.data.x.shape[1]
        self.pos_dim = self.data.pos.shape[1]
        self.dataset_name = 'rlmolecules'
        self.signal_class = 1
        self.feature_type = data_config['feature_type']
        

        node_categorical_feat = [len(self.ATOM_TYPES)]
        if self.feature_type == 'only_pos':
            node_scalar_feat = self.pos_dim
            node_categorical_feat = []
        elif self.feature_type == 'only_x':
            node_scalar_feat = self.x_dim - 1
        elif self.feature_type == 'only_ones':
            node_scalar_feat = 1
            node_categorical_feat = []
        else:
            assert self.feature_type == 'both_x_pos'
            node_scalar_feat = self.x_dim - 1 + self.pos_dim

        self.feat_info = {'node_categorical_feat': node_categorical_feat, 'node_scalar_feat': node_scalar_feat}
        

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):

        raw_idx_split = {
            "train_index": np.arange(0, int(0.8 * self.COUNT)),
            "test_index": np.arange(int(0.8 * self.COUNT), self.COUNT),
        }
        split_dict = self.get_split_dict(raw_idx_split, seed=self.seed)

        data_list = []
        idx_split = {"train": [], "valid": [], "test": []}

        mols = []
        class1_dir = 'rlmolecules_utils/ligands_pdb_class1'
        class0_dir = 'rlmolecules_utils/ligands_pdb_class0'
        

        class1_size = 0
        class0_size = 0
        for filename in os.listdir(class1_dir):
            if filename.endswith(".pdb"):  # Check for PDB files
                filepath = os.path.join(class1_dir, filename)
                mol = Chem.MolFromPDBFile(filepath, sanitize=False, removeHs=True)
                mols.append([mol, 1.0])
                class1_size += 1
                if class1_size == self.COUNT / 2:
                    break
        
        
        for filename in os.listdir(class0_dir):
            if filename.endswith(".pdb"):  # Check for PDB files
                filepath = os.path.join(class0_dir, filename)
                mol = Chem.MolFromPDBFile(filepath, sanitize=False, removeHs=True)
                mols.append([mol, 0.0])
                class0_size += 1
                if class0_size == self.COUNT / 2:
                    break
        
        random.seed(self.seed)
        random.shuffle(mols)

        print('class1_size:', class1_size, 'class0_size:', class0_size)

        n_node_features = 14
        
        element_to_index = {element: idx for idx, element in enumerate(self.ATOM_TYPES)}

        def one_hot_encode(element, elements):
            if element not in element_to_index:
                element = '*'
                print('wop')

            one_hot = torch.zeros(len(elements), dtype=torch.float)
            one_hot[element_to_index[element]] = 1.0
            return one_hot

        # Collect valid top and bottom molecules
        idx = 0
        for mol, y in mols:
            
            style='stick'
            mblock = Chem.MolToMolBlock(mol)
            view = py3Dmol.view(width=200, height=200)
            view.addModel(mblock, 'mol')
            view.setStyle({style:{}})
            view.zoomTo()
            view.show()
            
            
            n_nodes = mol.GetNumAtoms()
            x = np.zeros((n_nodes, n_node_features))
            for atom in mol.GetAtoms():
                x[atom.GetIdx(), :] = one_hot_encode(element = atom.GetSymbol(), elements = self.ATOM_TYPES)
        
            x = torch.tensor(x, dtype = torch.float)
            pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
            node_label = torch.randint(0, 2, (x.shape[0],), dtype=torch.long)

            y = torch.tensor(y).reshape(-1, 1)
            data = Data(x=x, pos=pos, y=y, node_label=node_label, mol_df_idx=idx)
            data_list.append(data)
            idx_split[split_dict[idx]].append(idx)
            idx += 1

        # Final data collation and saving
        self.data_list = data_list
        data, slices = self.collate(data_list)
        torch.save((data, slices, idx_split), self.processed_paths[0])

    @staticmethod
    def get_split_dict(raw_idx_split, seed):
        np.random.seed(seed)

        train_val_idx = raw_idx_split["train_index"]
        idx = np.arange(len(train_val_idx))
        np.random.shuffle(idx)

        N = len(train_val_idx)
        train_idx = train_val_idx[idx[:int(0.8 * N)]]
        valid_idx = train_val_idx[idx[int(0.8 * N):]]
        test_idx = raw_idx_split["test_index"]

        split_dict = {}
        for idx in train_idx:
            split_dict[idx] = "train"
        for idx in valid_idx:
            split_dict[idx] = "valid"
        for idx in test_idx:
            split_dict[idx] = "test"
        return split_dict


if __name__ == "__main__":
    data_config = yaml.safe_load(open('../configs/rlmolecules.yml'))['data']
    dataset = RLMolecules(root="../../data/rlmolecules", data_config=data_config, seed=42)
    data_list = dataset.data_list

    for mol in data_list:
        print(mol)




