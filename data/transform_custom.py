import json
import os
import numpy as np
import toml
from pathlib import Path

import pandas as pd


dir_path = Path(__file__).parent.resolve()

toml_file_name = dir_path / "custom_properties.toml"

with open(toml_file_name) as toml_file:
    custom_properties = toml.load(toml_file)

custom_atomic_num_list = custom_properties['atom_types'] + [0]
max_atoms = int(custom_properties['max_atoms'])
n_bonds = 4


def one_hot_custom(node, out_size=max_atoms):
    # node = atomic number array for each mol
    num_max_id = len(custom_atomic_num_list)
    # num_max_id = 1
    assert node.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = custom_atomic_num_list.index(node[i])
        b[i, ind] = 1.
    return b


def transform_fn(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_custom(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


#TODO: create a json index list of dataset elems used for testing
def get_val_ids():
    file_path = '../data/valid_idx_custom.json'
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids