import os
import sys

import toml
import json

# for linux env.
sys.path.insert(0,'..')
import pathlib
import pandas as pd
import numpy as np
from itertools import chain
import argparse
import time

from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import RDConfig

from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor


sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))

import sascorer


def get_num_atoms(smi_obj: str):
    atoms = Chem.MolFromSmiles(smi_obj).GetAtoms()
    x = [atoms[i].GetSymbol() for i in range(len(atoms))]
    return len(x)


def get_logp(smi_obj: str):
    return Crippen.MolLogP(Chem.MolFromSmiles(smi_obj))


def get_qed(smi_obj: str):
    return QED.qed(smi2mol(smi_obj))


def get_sas(smi_obj: str):
    return sascorer.calculateScore(Chem.MolFromSmiles(smi_obj))


def get_max_atoms(df: pd.DataFrame):
    if df.empty:
        df = pd.read_csv('custom.csv', index_col=0)

    df['num_atoms'] = df['smiles'].apply(lambda x: get_num_atoms(x))
    max_len = df['num_atoms'].max()
    return max_len


def get_df_atom_types(df: pd.DataFrame):
    df['z_atomic_nums'] = df['smiles'].apply(lambda x: get_atom_types(x))
    atom_types_mol = df['z_atomic_nums'].tolist()
    atom_types = list(set(list(chain.from_iterable(atom_types_mol))))
    atom_types.sort()
    return atom_types


def get_features():
    for label, cal in zip(['logP', 'qed', 'SAS'], [get_logp, get_qed, get_sas]):
        df_custom[label] = df_custom.apply(lambda x: get_logp(x[0]), axis=1)
        labels.append(label)
    df_custom.drop(columns=['num_atoms', 'z_atomic_nums'], inplace=True)


def get_atom_types(smi_obj: str):
    return [a.GetAtomicNum() for a in Chem.MolFromSmiles(smi_obj).GetAtoms()]


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='qm9',
                        choices=['qm9', 'zinc250k', 'custom'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    parser.add_argument('--features', action='store_true')
    parser.add_argument('--training_smi_file_path')
    args = parser.parse_args()
    return args


start_time = time.time()
args = parse()
data_name = args.data_name
data_type = args.data_type

if data_name == 'custom':
    features = args.features
    file_path = args.training_smi_file_path

    dir_path = pathlib.Path(__file__).parent.resolve()

    smiles = []
    labels = []

    df_custom = None
    atom_types = None

    with open(file_path, "r") as ins:
        for idx, line in enumerate(ins):
            smiles.append(line.split('\n')[0])
        df = pd.DataFrame(smiles, columns=['smiles'])
        df.to_csv('custom.csv')

print('args', vars(args))

if data_name == 'qm9':
    max_atoms = 9
elif data_name == 'zinc250k':
    max_atoms = 38
elif data_name == 'custom':
    df_custom = pd.read_csv('custom.csv', index_col=0)
    max_atoms = get_max_atoms(df_custom)
    atom_types = get_df_atom_types(df_custom)
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


if data_type == 'relgcn':
    # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'qm9':
    print('Preprocessing qm9 data:')
    df_qm9 = pd.read_csv('qm9.csv', index_col=0)
    labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
              'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES1')
    result = parser.parse(df_qm9, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'zinc250k':
    print('Preprocessing zinc250k data')
    # dataset = datasets.get_zinc250k(preprocessor)
    df_zinc250k = pd.read_csv('zinc250k.csv', index_col=0)
    # Caution: Not reasonable but used in used in chain_chemistry\datasets\zinc.py:
    # 'smiles' column contains '\n', need to remove it.
    # Here we do not remove \n, because it represents atom N with single bond
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_zinc250k, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'custom':
        print("RUNNING ...")
        get_features()
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_custom, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']

        custom_properties = {'max_atoms': max_atoms, 'atom_types': atom_types}
        toml_file_name = pathlib.Path("./custom_properties.toml")
        with open(toml_file_name, "w") as toml_file:
            toml.dump(custom_properties, toml_file)

        valid_idx = np.random.randint(len(dataset)-1, size=int(np.floor(len(dataset)*.2))).tolist()
        valid_idx_file = './valid_idx_custom.json'
        with open(valid_idx_file, 'w') as file:
            json.dump(valid_idx, file)
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

NumpyTupleDataset.save(os.path.join(data_dir, '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)), dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)) )
