import os
import csv
# from pubchempy import *
import numpy as np
import numbers
import h5py
import math
import pandas as pd
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *
import random
import pickle
import sys
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def is_not_float(string_list):
    """ from tCCN
    https://github.com/Lowpassfilter/tCNNS-Project/blob/master/data/preprocess.py
    """
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True


"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

folder = "data/"
#folder = ""
folder = "ap_data/" # ap: new

def load_drug_list():
    """ from tCCN
    https://github.com/Lowpassfilter/tCNNS-Project/blob/master/data/preprocess.py

    Load the downloaded GDSC drug metadata.
    """
    filename = folder + "Druglist.csv"
    # csvfile = open(filename, "rb") # ap: comment
    csvfile = open(filename, "r") # ap: new
    reader = csv.reader(csvfile)
    next(reader, None)  # skip first row (header)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs


def write_drug_cid():
    """ from tCCN
    https://github.com/Lowpassfilter/tCNNS-Project/blob/master/data/preprocess.py

    Create 2 csv table:
    1. pychem_cid.csv: 2-col table with (drug_name, cid) pairs
    2. unknow_drug_by_pychem.csv: list of drug names that were not found in PubChem
    """
    drugs = load_drug_list()
    drug_id = []
    datas = []
    # outputfile = open(folder + 'pychem_cid.csv', 'wb') # ap: comment
    outputfile = open(folder + 'pychem_cid.csv', 'w') # ap: new
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print(drug, cid)
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    # outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb') # ap: comment
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'w') # ap: new
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)
    outputfile.close() # ap: new


def cid_from_other_source():
    """ from tCCN
    https://github.com/Lowpassfilter/tCNNS-Project/blob/master/data/preprocess.py

    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict:
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {
        k: v
        for k, v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])
    }
    return drug_cid_dict


def load_cid_dict():
    """ from tCCN
    https://github.com/Lowpassfilter/tCNNS-Project/blob/master/data/preprocess.py
    
    Load a csv that contains drug meta with CID (PubChem IDs).
    """
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict


def download_smiles():
    """ from tCCN
    https://github.com/Lowpassfilter/tCNNS-Project/blob/master/data/preprocess.py
    
    Download SMILES from PubChem.
    """
    cids_dict = load_cid_dict()
    cids = [v for k, v in cids_dict.iteritems()]
    inv_cids_dict = {v: k for k, v in cids_dict.iteritems()}
    download(
        'CSV',
        folder + 'drug_smiles.csv',
        cids,
        operation='property/CanonicalSMILES,IsomericSMILES',
        overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()


"""
The following code will convert the SMILES format into onehot format (comment from tCNN)
The following code will convert the SMILES format into graph format (relevant comment)
"""

def atom_features(atom):
    """ (ap) Extract atom features and put into array. """
    # a1 = one_of_k_encoding_unk(atom.GetSymbol(), [
    #         'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
    #         'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
    #         'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
    #         'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    #     ])
    # a2 = one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a3 = one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a4 = one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # a5 = [atom.GetIsAromatic()]
    # arr = np.array(a1 + a2 + a3 + a4 + a5)
    # return arr
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
            'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co',
            'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
            'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(),
                                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    """ (ap) Convert SMILES to graph. """
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()  # num atoms in molecule

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()  # return a directed graph
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
    # (ap) How is edges list different from edge_index list??
    # It seems that len(edge_index) is twice the size of len(edges)
    return c_size, features, edge_index


def load_drug_smile():
    """
    (ap) drug_smiles.csv is a table [224, 4] with 223 drugs.
    Creates and returns the following:
    drug_dict:   dict (drug names, rows in which the drug appears)
    drug_smile:  list of drug SMILES
    smile_graph: dict (SMILES, [c_size, features, edge_index])
    """
    reader = csv.reader(open(folder + "drug_smiles.csv"))  # generated by download_smiles()
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]   # Drug name
        smile = item[2]  # Drug canonical SMILES

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)

    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)  # (ap) g: [c_size, features, edge_index]
        smile_graph[smile] = g

    return drug_dict, drug_smile, smile_graph


"""
The following part will prepare the mutation features for the cell.
"""

def save_cell_mut_matrix():
    """
    (ap) Create a binary matrix where 1 indicates that a mutation
    is present. Rows are CCLs and cols are mutations.
    PANCANCER_Genetic_feature.csv is a table [714056, 6]. The
    col "genetic_feature" contains either mutation suffixed with
    "_mut" or CNA prefixes with "cna_"
    """
    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}    # dict of CCL
    mut_dict = {}     # dict of genetic features (mutations and CNA)
    matrix_list = []  # list of matrix coordinates where mutations are present

    for item in reader:
        cell_id = item[1]          # CCL ID
        mut = item[5]              # mutation
        is_mutated = int(item[6])  # whether it's mutated

        # (ap) Mutations will be stored in columns
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        # (ap) CCLs will be stored in rows
        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row

        # (ap) append coordinates where mutations are active
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)

    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""

def save_mix_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv") # contains the IC50 of 250 drugs and 1074 CCL
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    with open('drug_dict', 'wb') as fp:
        pickle.dump(drug_dict, fp)

    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    temp_data = []
    for item in reader:
        drug = item[0]  # Drug name
        cell = item[3]  # Cosmic sample id
        ic50 = item[8]  # IC50
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xc = []
    y = []
    lst_drug = []
    lst_cell = []

    random.shuffle(temp_data)  # shuffle cell-drug combinations
    for data in temp_data:  # tuples of (drug name, cell id, IC50)
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:  # len(drug_dict): 223, len(cell_dict): 990
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)
            bExist[drug_dict[drug], cell_dict[cell]] = 1  # This is not used (??)
            lst_drug.append(drug)
            lst_cell.append(cell)

    # Three arrays of size 191049, as the number of responses
    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    # Define vars that determine train, val, and test sizes
    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    with open('list_drug_mix_test', 'wb') as fp:
        pickle.dump(lst_drug[size1:], fp)

    with open('list_cell_mix_test', 'wb') as fp:
        pickle.dump(lst_cell[size1:], fp)

    # Create data splits
    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]

    xc_train = xc[:size]
    xc_val = xc[size:size1]
    xc_test = xc[size1:]

    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]

    # Create PyTorch datasets
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        root='data',
        dataset=dataset + '_train_mix',
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        root='data',
        dataset=dataset + '_val_mix',
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        root='data',
        dataset=dataset + '_test_mix',
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)


def save_blind_drug_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []  # not used
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    temp_data = []
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []  # not used
    xc_unknown = []  # not used
    y_unknown = []   # not used

    dict_drug_cell = {}

    random.shuffle(temp_data)  # shuffle cell-drug combinations
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]

            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstDrugTest = []

    # Define vars that determine train, val, and test sizes
    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)

    # Create data splits
    pos = 0
    for drug, values in dict_drug_cell.items():
        pos += 1
        for v in values:
            cell, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstDrugTest.append(drug)

    with open('drug_blind_test', 'wb') as fp:
        pickle.dump(lstDrugTest, fp)

    print(len(y_train), len(y_val), len(y_test))
    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    # Create PyTorch datasets
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        root='data',
        dataset=dataset + '_train_blind',
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        root='data',
        dataset=dataset + '_val_blind',
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        root='data',
        dataset=dataset + '_test_blind',
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)


def save_blind_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []  # not used
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    temp_data = []
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd_train = []
    xc_train = []
    y_train = []

    xd_val = []
    xc_val = []
    y_val = []

    xd_test = []
    xc_test = []
    y_test = []

    xd_unknown = []  # not used
    xc_unknown = []  # not used
    y_unknown = []   # not used

    dict_drug_cell = {}

    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((drug, ic50))
            else:
                dict_drug_cell[cell] = [(drug, ic50)]

            bExist[drug_dict[drug], cell_dict[cell]] = 1

    lstCellTest = []

    # Define vars that determine train, val, and test sizes
    size = int(len(dict_drug_cell) * 0.8)
    size1 = int(len(dict_drug_cell) * 0.9)

    # Create data splits
    pos = 0
    for cell, values in dict_drug_cell.items():
        pos += 1
        for v in values:
            drug, ic50 = v
            if pos < size:
                xd_train.append(drug_smile[drug_dict[drug]])
                xc_train.append(cell_feature[cell_dict[cell]])
                y_train.append(ic50)
            elif pos < size1:
                xd_val.append(drug_smile[drug_dict[drug]])
                xc_val.append(cell_feature[cell_dict[cell]])
                y_val.append(ic50)
            else:
                xd_test.append(drug_smile[drug_dict[drug]])
                xc_test.append(cell_feature[cell_dict[cell]])
                y_test.append(ic50)
                lstCellTest.append(cell)

    with open('cell_blind_test', 'wb') as fp:
        pickle.dump(lstCellTest, fp)

    print(len(y_train), len(y_val), len(y_test))
    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    # Create PyTorch datasets
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        root='data',
        dataset=dataset + '_train_cell_blind',
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        root='data',
        dataset=dataset + '_val_cell_blind',
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        root='data',
        dataset=dataset + '_test_cell_blind',
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)


def save_best_individual_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    matrix_list = []  # not used
    bExist = np.zeros((len(drug_dict), len(cell_dict)))

    temp_data = []
    i = 0
    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        if drug == "Bortezomib":
            temp_data.append((drug, cell, ic50))

    xd_train = []
    xc_train = []
    y_train = []

    dict_drug_cell = {}

    random.shuffle(temp_data)
    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]

            bExist[drug_dict[drug], cell_dict[cell]] = 1
    cells = []
    for drug, values in dict_drug_cell.items():
        for v in values:
            cell, ic50 = v
            xd_train.append(drug_smile[drug_dict[drug]])
            xc_train.append(cell_feature[cell_dict[cell]])
            y_train.append(ic50)
            cells.append(cell)

    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)

    with open('cell_blind_sal', 'wb') as fp:
        pickle.dump(cells, fp)

    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        root='data',
        dataset=dataset + '_bortezomib',
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph,
        saliency_map=True)


def create_complete_set():
    """ Create table that lists of [drug, cell, ic50] that have all the
    required features.
    """


def create_lc_sets(split_type: str="mix"):
    """ Creates subsets for training, val, and test. """
    f = open(folder + "PANCANCER_IC.csv") # IC50 of 250 drugs and 1074 CCL
    reader = csv.reader(f)
    next(reader)

    cell_dict, _ = save_cell_mut_matrix()
    drug_dict, _, _ = load_drug_smile()

    temp_data = []
    for item in reader:
        drug = item[0]  # Drug name
        cell = item[3]  # Cosmic sample id
        ic50 = item[8]  # IC50
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    fdir = os.path.dirname(os.path.abspath(__file__))
    lc_dir = Path(os.path.join(fdir, "lc_data"))
    os.makedirs(lc_dir, exist_ok=True)

    import pdb; pdb.set_trace()
    if split_type == "mix":
        outdir = lc_dir/"mix_drug_cell"
        os.makedirs(outdir, exist_ok=True)

        drug_cell_rsp = []  # ap: data that contains all features
        random.shuffle(temp_data)  # shuffle cell-drug combinations
        for data in temp_data:  # tuples of (drug name, cell id, IC50)
            drug, cell, ic50 = data
            if drug in drug_dict and cell in cell_dict:
                drug_cell_rsp.append((drug, cell, ic50))  # ap

        df = pd.DataFrame(drug_cell_rsp, columns=["Drug", "Cell", "IC50"])

        # Define vars that determine train, val, and test sizes
        size = int(len(df) * 0.8)
        size1 = int(len(df) * 0.9)

        df_tr = df[:size]
        df_vl = df[size:size1]
        df_te = df[size1:]
        df_tr.to_csv(outdir/"train_data.csv", index=False)
        df_vl.to_csv(outdir/"val_data.csv", index=False)
        df_te.to_csv(outdir/"test_data.csv", index=False)

        df = pd.concat([df_tr, df_vl, df_te], axis=0)
        df.to_csv(outdir/"drug_cell_rsp.csv", index=False)

    elif split_type == "drug":
        outdir = lc_dir/"drug_blind"
        os.makedirs(outdir, exist_ok=True)

        dict_drug_cell = {}
        random.shuffle(temp_data)  # shuffle cell-drug combinations
        for data in temp_data:
            drug, cell, ic50 = data
            if drug in drug_dict and cell in cell_dict:
                if drug in dict_drug_cell:
                    dict_drug_cell[drug].append((cell, ic50))
                else:
                    dict_drug_cell[drug] = [(cell, ic50)]

        # Define vars that determine train, val, and test sizes
        size = int(len(dict_drug_cell) * 0.8)
        size1 = int(len(dict_drug_cell) * 0.9)

        import pdb; pdb.set_trace()
        # Create data splits
        df_tr = []  # ap: data that contains all features
        df_vl = []  # ap: data that contains all features
        df_te = []  # ap: data that contains all features
        pos = 0
        for drug, values in dict_drug_cell.items():
            pos += 1
            for v in values:
                cell, ic50 = v
                if pos < size:
                    df_tr.append((drug, cell, ic50))  # ap
                elif pos < size1:
                    df_vl.append((drug, cell, ic50))  # ap
                else:
                    df_te.append((drug, cell, ic50))  # ap

        df_tr = pd.DataFrame(df_tr, columns=["Drug", "Cell", "IC50"])
        df_vl = pd.DataFrame(df_vl, columns=["Drug", "Cell", "IC50"])
        df_te = pd.DataFrame(df_te, columns=["Drug", "Cell", "IC50"])
        df_tr.to_csv(outdir/"train_data.csv", index=False)
        df_vl.to_csv(outdir/"val_data.csv", index=False)
        df_te.to_csv(outdir/"test_data.csv", index=False)

        # set(df_tr["Drug"].values).intersection(set(df_vl["Drug"].values))
        # set(df_tr["Drug"].values).intersection(set(df_te["Drug"].values))
        # set(df_vl["Drug"].values).intersection(set(df_te["Drug"].values))

        df = pd.concat([df_tr, df_vl, df_te], axis=0)
        df.to_csv(outdir/"drug_cell_rsp.csv", index=False)

    elif split_type == "cell":
        outdir = lc_dir/"cell_blind"
        os.makedirs(outdir, exist_ok=True)
        
        df_tr = None
        df_vl = None
        df_te = None

    # --------------------------------------------

    # Save splits
    # lc_init_args = {'cv_lists': cv_lists,
    #                 'lc_step_scale': args['lc_step_scale'],
    #                 'lc_sizes': args['lc_sizes'],
    #                 'min_size': args['min_size'],
    #                 'max_size': args['max_size'],
    #                 'lc_sizes_arr': args['lc_sizes_arr'],
    #                 'print_fn': print
    #                 }

    lc_init_args = {'cv_lists': None,
                    'lc_step_scale': "log",
                    'lc_sizes': 7,
                    'min_size': None,
                    'max_size': None,
                    'lc_sizes_arr': None,
                    'print_fn': print
                    }

    lc_step_scale = "log"
    lc_sizes = 7
    min_size = 1024
    max_size = df_tr.shape[0]
    # from learningcurve.lrn_crv import LearningCurve
    # lc_obj = LearningCurve(X=None, Y=None, meta=None, **lc_init_args)
    pw = np.linspace(0, lc_sizes-1, num=lc_sizes) / (lc_sizes-1)
    m = min_size * (max_size/min_size) ** pw
    m = np.array([int(i) for i in m])  # cast to int
    lc_sizes = m

    # LC subsets
    for i, sz in enumerate(lc_sizes):
        aa = df_tr[:sz]
        aa.to_csv(outdir/f"train_sz_{i+1}.csv", index=False)
    return None


def create_arr_from_data(df, drug_dict, cell_dict, drug_smile, cell_feature):
    """ ... """
    xd, xc, y = [], [], []

    # import pdb; pdb.set_trace()
    df_data = [(r[1]["Drug"], r[1]["Cell"], r[1]["IC50"]) for r in df.iterrows()]
    for data in df_data:  # tuples of (drug name, cell id, IC50)
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            xd.append(drug_smile[drug_dict[drug]])
            xc.append(cell_feature[cell_dict[cell]])
            y.append(ic50)

    # Three arrays of size 191049, as the number of responses
    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
    return xd, xc, y


def create_lc_datasets(split_type: str="mix"):
    """ Creates subsets for training, val, and test. """
    f = open(folder + "PANCANCER_IC.csv") # IC50 of 250 drugs and 1074 CCL
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    fdir = os.path.dirname(os.path.abspath(__file__))
    lc_dir = Path(os.path.join(fdir, "lc_data"))

    import pdb; pdb.set_trace()
    if split_type == "mix":
        datadir = lc_dir/"mix_drug_cell"
        types = {"Drug": str, "Cell": str, "IC50": float}

        # Create PyTorch datasets
        dataset = 'GDSC'
        print('preparing ', dataset + '_train.pt in pytorch format!')
        root = datadir

        fn_args = {"drug_dict": drug_dict,
                   "cell_dict": cell_dict,
                   "drug_smile": drug_smile,
                   "cell_feature": cell_feature}

        tr_sz_files = list(datadir.glob("train_sz*.csv"))
        for fpath in tr_sz_files:
            data = pd.read_csv(fpath, dtype=types)
            xd_train, xc_train, y_train = create_arr_from_data(data, **fn_args)
            train_data = TestbedDataset(
                root=root,
                dataset=fpath.with_suffix("").name,
                xd=xd_train,
                xt=xc_train,
                y=y_train,
                smile_graph=smile_graph)

        tr_data = pd.read_csv(datadir/"train_data.csv", dtype=types)
        xd_train, xc_train, y_train = create_arr_from_data(tr_data, **fn_args)
        val_data = TestbedDataset(
            root=root,
            dataset="train_data",
            xd=xd_train,
            xt=xc_train,
            y=y_train,
            smile_graph=smile_graph)

        vl_data = pd.read_csv(datadir/"val_data.csv", dtype=types)
        xd_val, xc_val, y_val = create_arr_from_data(vl_data, **fn_args)
        val_data = TestbedDataset(
            root=root,
            dataset="val_data",
            xd=xd_val,
            xt=xc_val,
            y=y_val,
            smile_graph=smile_graph)

        te_data = pd.read_csv(datadir/"test_data.csv", dtype=types)
        xd_test, xc_test, y_test = create_arr_from_data(
            te_data, drug_dict, cell_dict, drug_smile, cell_feature)
        test_data = TestbedDataset(
            root=root,
            dataset="test_data",
            xd=xd_test,
            xt=xc_test,
            y=y_test,
            smile_graph=smile_graph)

    elif split_type == "drug":
        datadir = lc_dir/"drug_blind"

    elif split_type == "cell":
        datadir = lc_dir/"cell_blind"
    return None
        

if __name__ == "__main__":
    import candle
    from pathlib import Path
    fdir = Path(__file__).resolve().parent

    ftp_fname = fdir/"ftp_file_list"
    with open(ftp_fname, "r") as f:
        data_file_list = f.readlines()

    ftp_origin = "https://ftp.mcs.anl.gov/pub/candle/public/improve/reproducability/GraphDRP/data"
    for f in data_file_list:
        candle.get_file(fname=f.strip(),
                        origin=os.path.join(ftp_origin, f.strip()),
                        unpack=False, md5_hash=None,
                        datadir=fdir/"./data",
                        cache_subdir="common")

    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument(
        '--choice',
        type=int,
        required=False,
        default=0,
        help='0.mix test, 1.saliency value, 2.drug blind, 3.cell blind')
    args = parser.parse_args()
    choice = args.choice

    # import pdb; pdb.set_trace()
    create_lc_sets(split_type="mix")
    # create_lc_sets(split_type="drug")

    # import pdb; pdb.set_trace()
    create_lc_datasets(split_type="mix")

#     import pdb; pdb.set_trace()
#     if choice == 0:
#         # save mix test dataset
#         save_mix_drug_cell_matrix()
#     elif choice == 1:
#         # save saliency map dataset
#         save_best_individual_drug_cell_matrix()
#     elif choice == 2:
#         # save blind drug dataset
#         save_blind_drug_matrix()
#     elif choice == 3:
#         # save blind cell dataset
#         save_blind_cell_matrix()
#     else:
#         print("Invalide option, choose 0 -> 4")
