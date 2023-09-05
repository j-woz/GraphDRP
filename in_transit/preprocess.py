from collections import OrderedDict
from pathlib import Path
from pubchempy import *
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from utils import *
import argparse
import candle
import csv
import h5py
import json, pickle
import math
import matplotlib.pyplot as plt
import networkx as nx
import numbers
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys


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
fdir = os.path.dirname(os.path.abspath(__file__)) # parent dir


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
    Create a binary matrix where 1 indicates that a mutation is present.
    Rows are CCLs and cols are mutations.
    PANCANCER_Genetic_feature.csv is a table [714056, 6].
    The col "genetic_feature" contains either mutation suffixed with
    "_mut" or CNA prefixes with "cna_"
    """
    # aa = pd.read_csv(folder + "PANCANCER_Genetic_feature.csv")
    # print("Read PANCANCER_Genetic_feature.csv")
    # print(aa.shape)
    # print(aa[:2])

    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}    # dict of CCL
    mut_dict = {}     # dict of genetic features (mutations and CNA)
    matrix_list = []  # list of matrix coordinates where mutations are present

    for item in reader:
        cell_id = item[1]          # CCL ID (cosmic_sample_id)
        mut = item[5]              # mutation (genetic_feature)
        is_mutated = int(item[6])  # whether it's mutated (is_mutated)

        # Mutations will be stored in columns
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        # CCLs will be stored in rows
        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row

        # Append coordinates where mutations are active
        if is_mutated == 1:
            matrix_list.append((row, col))

    # Create 2-D array [cells, mutations]
    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    # Iterate over a list of (cell, genes) tuples and assign 1 for mutated genes
    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    # with open('mut_dict', 'wb') as fp:
    #     pickle.dump(mut_dict, fp)

    return cell_dict, cell_feature


"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score
"""

def save_mix_drug_cell_matrix(args):
    f = open(folder + "PANCANCER_IC.csv") # contains the IC50 of 250 drugs and 1074 CCL
    reader = csv.reader(f)
    next(reader)

    root = os.path.join(args.outdir, "mixed_set")

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    # with open('drug_dict', 'wb') as fp:
    #     pickle.dump(drug_dict, fp)

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
            lst_drug.append(drug)
            lst_cell.append(cell)
            bExist[drug_dict[drug], cell_dict[cell]] = 1  # not used

    # Three arrays of size 191049, as the number of responses
    xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)

    # Define vars that determine train, val, and test sizes
    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    # with open('list_drug_mix_test', 'wb') as fp:
    #     pickle.dump(lst_drug[size1:], fp)

    # with open('list_cell_mix_test', 'wb') as fp:
    #     pickle.dump(lst_cell[size1:], fp)

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
    # dataset = 'GDSC'
    dataset = "data"
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_train_mix',
        dataset='train_' + dataset,
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_val_mix',
        dataset='val_' + dataset,
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_test_mix',
        dataset='test_' + dataset,
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)


def save_blind_drug_matrix(args):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    root = os.path.join(args.outdir, "drug_blind")

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    # with open('drug_dict', 'wb') as fp:
    #     pickle.dump(drug_dict, fp)

    # matrix_list = []  # not used
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

    dict_drug_cell = {}  # keys are drug names

    random.shuffle(temp_data)  # shuffle cell-drug combinations
    for data in temp_data:  # tuples of (drug name, cell id, IC50)
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if drug in dict_drug_cell:
                dict_drug_cell[drug].append((cell, ic50))
            else:
                dict_drug_cell[drug] = [(cell, ic50)]
            bExist[drug_dict[drug], cell_dict[cell]] = 1  # not used

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

    # with open('drug_blind_test', 'wb') as fp:
    #     pickle.dump(lstDrugTest, fp)

    print(len(y_train), len(y_val), len(y_test))
    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    # Create PyTorch datasets
    # dataset = 'GDSC'
    dataset = "data"
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_train_blind',
        dataset='train_' + dataset,
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_val_blind',
        dataset='val_' + dataset,
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_test_blind',
        dataset='test_' + dataset,
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)


def save_blind_cell_matrix(args):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    root = os.path.join(args.outdir, "cell_blind")

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    # with open('drug_dict', 'wb') as fp:
    #     pickle.dump(drug_dict, fp)

    # matrix_list = []  # not used
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

    dict_drug_cell = {}  # keys are cell names

    random.shuffle(temp_data)  # shuffle cell-drug combinations
    for data in temp_data:  # tuples of (drug name, cell id, IC50)
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict:
            if cell in dict_drug_cell:
                dict_drug_cell[cell].append((drug, ic50))
            else:
                dict_drug_cell[cell] = [(drug, ic50)]
            bExist[drug_dict[drug], cell_dict[cell]] = 1  # not used

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

    # with open('cell_blind_test', 'wb') as fp:
    #     pickle.dump(lstCellTest, fp)

    print(len(y_train), len(y_val), len(y_test))
    xd_train, xc_train, y_train = np.asarray(xd_train), np.asarray(xc_train), np.asarray(y_train)
    xd_val, xc_val, y_val = np.asarray(xd_val), np.asarray(xc_val), np.asarray(y_val)
    xd_test, xc_test, y_test = np.asarray(xd_test), np.asarray(xc_test), np.asarray(y_test)

    # Create PyTorch datasets
    # dataset = 'GDSC'
    dataset = "data"
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_train_cell_blind',
        dataset="train_" + dataset,
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_val_cell_blind',
        dataset="val_" + dataset,
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        # root='data',
        root=root,
        # dataset=dataset + '_test_cell_blind',
        dataset="test_" + dataset,
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)


def save_best_individual_drug_cell_matrix(args):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict, cell_feature = save_cell_mut_matrix()
    drug_dict, drug_smile, smile_graph = load_drug_smile()

    # matrix_list = []  # not used
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
            bExist[drug_dict[drug], cell_dict[cell]] = 1  # not used

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


"""
The functions below generate datasets for CSG (data from July 2020) - Start
"""


def read_df(fpath, sep="\t"):
    assert Path(fpath).exists(), f"File {fpath} was not found."
    if "parquet" in str(fpath):
        df = pd.read_parquet(fpath)
    else:
        df = pd.read_csv(fpath, sep=sep, na_values=na_values)
    return df


def scale_fea(xdata, scaler_name='stnd', dtype=np.float32, verbose=False):
    """ Returns the scaled dataframe of features. """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    if scaler_name is None:
        if verbose:
            print('Scaler is None (not scaling).')
        return xdata
    
    if scaler_name == 'stnd':
        scaler = StandardScaler()
    elif scaler_name == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_name == 'rbst':
        scaler = RobustScaler()
    else:
        print(f'The specified scaler {scaler_name} is not supported (not scaling).')
        return xdata

    cols = xdata.columns
    return pd.DataFrame(scaler.fit_transform(xdata), columns=cols, dtype=dtype)


def gen_csg_data(args, csg_datadir):

    sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]

    for src in sources:

        src_datadir = Path(csg_datadir)/f"data.{src}"
        root = Path(src_datadir)/f"data_split_{args.split}"
        os.makedirs(root, exist_ok=True)

        # -------------------
        # Response
        pathlist = list(Path(src_datadir).glob("rsp*.csv"))
        pathlist = [p for p in pathlist if "full" not in str(p)]
        rsp_df = pd.read_csv(pathlist[0])
        rsp_df = rsp_df[["DrugID", "CancID", "AUC"]]  # temp_data
        print(rsp_df[["CancID", "DrugID"]].nunique())

        # Drugs
        pathlist = list(Path(src_datadir).glob("smiles*.csv"))
        smi = pd.read_csv(pathlist[0])
        d_dict = {v: i for i, v in enumerate(smi["DrugID"].values)}  # drug_dict; len(d_dict): 311
        d_smile = smi["SMILES"].values  # drug_smile
        smile_graph = {}  # smile_graph
        dd = {d_id: s for d_id, s in zip(smi["DrugID"].values, smi["SMILES"].values)}
        for smile in d_smile:
            g = smile_to_graph(smile)  # (ap) g: [c_size, features, edge_index]
            smile_graph[smile] = g

        print("Unique drugs: {}".format(len(d_dict)))
        print("Unique smiles: {}".format(len(smile_graph)))

        # Cells
        pathlist = list(Path(src_datadir).glob("ge*.parquet"))
        ge = read_df(pathlist[0])

        # Use landmark genes
        use_lincs = True
        if use_lincs:
            with open(Path(src_datadir)/"../landmark_genes") as f:
                genes = [str(line.rstrip()) for line in f]
            genes = ["ge_" + str(g) for g in genes]
            print("Genes count: {}".format(len(set(genes).intersection(set(ge.columns[1:])))))
            genes = list(set(genes).intersection(set(ge.columns[1:])))
            cols = ["CancID"] + genes
            ge = ge[cols]

        # Scale
        ge_xdata = ge.iloc[:, 1:]
        ge_xdata_scaled = scale_fea(ge_xdata, scaler_name='stnd', dtype=np.float32, verbose=False)
        ge = pd.concat([ge[["CancID"]], ge_xdata_scaled], axis=1)

        # ge = ge.iloc[:, :1000]  # Take subset of cols (genes)
        c_dict = {v: i for i, v in enumerate(ge["CancID"].values)}  # cell_dict; len(c_dict): 634
        c_feature = ge.iloc[:, 1:].values  # cell_feature
        cc = {c_id: ge.iloc[i, 1:].values for i, c_id in enumerate(ge["CancID"].values)}  # cell_dict; len(c_dict): 634

        # Data splits
        splitdir = Path(os.path.join(src_datadir))/"splits"
        with open(splitdir/f'split_{args.split}_tr_id') as f:
            tr_id = [int(line.rstrip()) for line in f]
        with open(splitdir/f'split_{args.split}_te_id') as f:
            te_id = [int(line.rstrip()) for line in f]

        # Train and test data
        tr_data = rsp_df.loc[tr_id]
        te_data = rsp_df.loc[te_id]

        # Val data from tr_data
        from sklearn.model_selection import train_test_split
        tr_id, vl_id = train_test_split(tr_id, test_size=0.11)
        tr_data = rsp_df.loc[tr_id]
        vl_data = rsp_df.loc[vl_id]
        print("All  ", rsp_df.shape)
        print("Train", tr_data.shape)
        print("Val  ", vl_data.shape)
        print("Test ", te_data.shape)
        # del rsp_df

        def extract_data_vars(df, d_dict, c_dict, d_smile, c_feature, dd, cc):
            xd = []
            xc = []
            y = []
            xd_ = []
            xc_ = []
            nan_rsp_list = []
            miss_cell = []
            miss_drug = []
            meta = []
            # import ipdb; ipdb.set_trace()
            for i in range(df.shape[0]):  # tuples of (drug name, cell id, IC50)
                if i>0 and (i%15000 == 0):
                    print(i)
                drug, cell, rsp = df.iloc[i, :].values.tolist()
                if np.isnan(rsp):
                    nan_rsp_list.append(data)
                # If drug and cell features are available
                if drug in d_dict and cell in c_dict:  # len(drug_dict): 223, len(cell_dict): 990
                    xd.append(d_smile[d_dict[drug]])   # xd contains list of smiles
                    # xd_.append(dd[drug])   # xd contains list of smiles
                    xc.append(c_feature[c_dict[cell]]) # xc contains list of feature vectors
                    # xc_.append(cc[cell]) # xc contains list of feature vectors
                    y.append(rsp)
                    meta.append([drug, cell, rsp])
                elif cell not in c_dict:
                    import ipdb; ipdb.set_trace()
                    miss_cell.append(cell)
                elif drug not in d_dict:
                    import ipdb; ipdb.set_trace()
                    miss_drug.append(drug)

            # Three arrays of size 191049, as the number of responses
            xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
            xd_, xc_ = np.asarray(xd_), np.asarray(xc_)
            meta = pd.DataFrame(meta, columns=["DrugID", "CancID", "AUC"])

            return xd, xc, y

        xd_all,   xc_all,   y_all   = extract_data_vars(rsp_df,  d_dict, c_dict, d_smile, c_feature, dd, cc)
        xd_train, xc_train, y_train = np.take(xd_all, tr_id, axis=0), np.take(xc_all, tr_id, axis=0), np.take(y_all, tr_id, axis=0)
        xd_val,   xc_val,   y_val   = np.take(xd_all, vl_id, axis=0), np.take(xc_all, vl_id, axis=0), np.take(y_all, vl_id, axis=0)
        xd_test,  xc_test,  y_test  = np.take(xd_all, te_id, axis=0), np.take(xc_all, te_id, axis=0), np.take(y_all, te_id, axis=0)

        print("xd_all  ", xd_all.shape,   "xc_all  ", xc_all.shape,   "y_all  ", y_all.shape)
        print("xd_train", xd_train.shape, "xc_train", xc_train.shape, "y_train", y_train.shape)
        print("xd_val  ", xd_val.shape,   "xc_val  ", xc_val.shape,   "y_val  ", y_val.shape)
        print("xd_test ", xd_test.shape,  "xc_test ", xc_test.shape,  "y_test ", y_test.shape)

        # Save dfs
        rsp_df.to_csv(root/"all_rsp.csv", index=False)
        tr_data.to_csv(root/"train_rsp.csv", index=False)
        vl_data.to_csv(root/"val_rsp.csv", index=False)
        te_data.to_csv(root/"test_rsp.csv", index=False)
        del rsp_df, tr_data, vl_data, te_data
        # -------------------

        # Create PyTorch datasets
        dataset = "data"
        print('preparing ', dataset + '_train.pt in pytorch format!')

        # Train, val, and test datasets
        train_data = TestbedDataset(
            root=root,
            dataset='train_' + dataset,
            xd=xd_train,
            xt=xc_train,
            y=y_train,
            smile_graph=smile_graph)
        val_data = TestbedDataset(
            root=root,
            dataset='val_' + dataset,
            xd=xd_val,
            xt=xc_val,
            y=y_val,
            smile_graph=smile_graph)
        test_data = TestbedDataset(
            root=root,
            dataset='test_' + dataset,
            xd=xd_test,
            xt=xc_test,
            y=y_test,
            smile_graph=smile_graph)

        # All samples dataset
        all_data = TestbedDataset(
            root=root,
            dataset='all_' + dataset,
            xd=xd_all,
            xt=xc_all,
            y=y_all,
            smile_graph=smile_graph)


def raw_drp_to_ml_data(args, raw_drp_datadir):
    """ Generate a single ML data file from raw DRP data. The raw DRP data is
    defined as IMPROVE doc website. """

    import ipdb; ipdb.set_trace()
    # sources = ["ccle", "ctrp", "gcsi", "gdsc1", "gdsc2"]
    src = "gdsc2"

    # for src in sources:

    # src_datadir = Path(raw_drp_datadir)/f"data.{src}"
    # root = Path(raw_drp_datadir)/f"data_split_{args.split}"
    root = fdir/"ml_data"/f"data_split_{args.split}" # ML data
    os.makedirs(root, exist_ok=True)

    # -------------------
    # Response
    pathlist = list(Path(raw_drp_datadir).glob("rsp*.csv"))
    pathlist = [p for p in pathlist if "full" not in str(p)]
    rsp_df = pd.read_csv(pathlist[0])
    rsp_df = rsp_df[["DrugID", "CancID", "AUC"]]  # temp_data
    print(rsp_df[["CancID", "DrugID"]].nunique())

    # Drugs
    pathlist = list(Path(raw_drp_datadir).glob("smiles*.csv"))
    smi = pd.read_csv(pathlist[0])
    d_dict = {v: i for i, v in enumerate(smi["DrugID"].values)}  # drug_dict; len(d_dict): 311
    d_smile = smi["SMILES"].values  # drug_smile
    smile_graph = {}  # smile_graph
    dd = {d_id: s for d_id, s in zip(smi["DrugID"].values, smi["SMILES"].values)}
    for smile in d_smile:
        g = smile_to_graph(smile)  # (ap) g: [c_size, features, edge_index]
        smile_graph[smile] = g

    print("Unique drugs: {}".format(len(d_dict)))
    print("Unique smiles: {}".format(len(smile_graph)))

    # Cells
    pathlist = list(Path(raw_drp_datadir).glob("ge*.parquet"))
    ge = read_df(pathlist[0])

    # Use landmark genes
    use_lincs = True
    if use_lincs:
        with open(Path(raw_drp_datadir)/"../landmark_genes") as f:
            genes = [str(line.rstrip()) for line in f]
        genes = ["ge_" + str(g) for g in genes]
        print("Genes count: {}".format(len(set(genes).intersection(set(ge.columns[1:])))))
        genes = list(set(genes).intersection(set(ge.columns[1:])))
        cols = ["CancID"] + genes
        ge = ge[cols]

    # Scale
    ge_xdata = ge.iloc[:, 1:]
    ge_xdata_scaled = scale_fea(ge_xdata, scaler_name='stnd', dtype=np.float32, verbose=False)
    ge = pd.concat([ge[["CancID"]], ge_xdata_scaled], axis=1)

    # ge = ge.iloc[:, :1000]  # Take subset of cols (genes)
    c_dict = {v: i for i, v in enumerate(ge["CancID"].values)}  # cell_dict; len(c_dict): 634
    c_feature = ge.iloc[:, 1:].values  # cell_feature
    cc = {c_id: ge.iloc[i, 1:].values for i, c_id in enumerate(ge["CancID"].values)}  # cell_dict; len(c_dict): 634

    # Data splits
    splitdir = Path(os.path.join(raw_drp_datadir))/"splits"
    with open(splitdir/f'split_{args.split}_tr_id') as f:
        tr_id = [int(line.rstrip()) for line in f]
    with open(splitdir/f'split_{args.split}_vl_id') as f:
        vl_id = [int(line.rstrip()) for line in f]
    with open(splitdir/f'split_{args.split}_te_id') as f:
        te_id = [int(line.rstrip()) for line in f]

    # Train and test data
    tr_data = rsp_df.loc[tr_id]
    vl_data = rsp_df.loc[vl_id]
    te_data = rsp_df.loc[te_id]

    # Val data from tr_data
    # from sklearn.model_selection import train_test_split
    # tr_id, vl_id = train_test_split(tr_id, test_size=0.11)
    # tr_data = rsp_df.loc[tr_id]
    # vl_data = rsp_df.loc[vl_id]
    print("All  ", rsp_df.shape)
    print("Train", tr_data.shape)
    print("Val  ", vl_data.shape)
    print("Test ", te_data.shape)
    # del rsp_df

    def extract_data_vars(df, d_dict, c_dict, d_smile, c_feature, dd, cc):
        xd = []
        xc = []
        y = []
        xd_ = []
        xc_ = []
        nan_rsp_list = []
        miss_cell = []
        miss_drug = []
        meta = []
        # import ipdb; ipdb.set_trace()
        for i in range(df.shape[0]):  # tuples of (drug name, cell id, IC50)
            if i>0 and (i%15000 == 0):
                print(i)
            drug, cell, rsp = df.iloc[i, :].values.tolist()
            if np.isnan(rsp):
                nan_rsp_list.append(data)
            # If drug and cell features are available
            if drug in d_dict and cell in c_dict:  # len(drug_dict): 223, len(cell_dict): 990
                xd.append(d_smile[d_dict[drug]])   # xd contains list of smiles
                # xd_.append(dd[drug])   # xd contains list of smiles
                xc.append(c_feature[c_dict[cell]]) # xc contains list of feature vectors
                # xc_.append(cc[cell]) # xc contains list of feature vectors
                y.append(rsp)
                meta.append([drug, cell, rsp])
            elif cell not in c_dict:
                import ipdb; ipdb.set_trace()
                miss_cell.append(cell)
            elif drug not in d_dict:
                import ipdb; ipdb.set_trace()
                miss_drug.append(drug)

        # Three arrays of size 191049, as the number of responses
        xd, xc, y = np.asarray(xd), np.asarray(xc), np.asarray(y)
        xd_, xc_ = np.asarray(xd_), np.asarray(xc_)
        meta = pd.DataFrame(meta, columns=["DrugID", "CancID", "AUC"])

        return xd, xc, y

    xd_all,   xc_all,   y_all   = extract_data_vars(rsp_df,  d_dict, c_dict, d_smile, c_feature, dd, cc)
    xd_train, xc_train, y_train = np.take(xd_all, tr_id, axis=0), np.take(xc_all, tr_id, axis=0), np.take(y_all, tr_id, axis=0)
    xd_val,   xc_val,   y_val   = np.take(xd_all, vl_id, axis=0), np.take(xc_all, vl_id, axis=0), np.take(y_all, vl_id, axis=0)
    xd_test,  xc_test,  y_test  = np.take(xd_all, te_id, axis=0), np.take(xc_all, te_id, axis=0), np.take(y_all, te_id, axis=0)

    print("xd_all  ", xd_all.shape,   "xc_all  ", xc_all.shape,   "y_all  ", y_all.shape)
    print("xd_train", xd_train.shape, "xc_train", xc_train.shape, "y_train", y_train.shape)
    print("xd_val  ", xd_val.shape,   "xc_val  ", xc_val.shape,   "y_val  ", y_val.shape)
    print("xd_test ", xd_test.shape,  "xc_test ", xc_test.shape,  "y_test ", y_test.shape)

    # Save dfs
    rsp_df.to_csv(root/"all_rsp.csv", index=False)
    tr_data.to_csv(root/"train_rsp.csv", index=False)
    vl_data.to_csv(root/"val_rsp.csv", index=False)
    te_data.to_csv(root/"test_rsp.csv", index=False)
    del rsp_df, tr_data, vl_data, te_data
    # -------------------

    # Create PyTorch datasets
    dataset = "data"
    print('preparing ', dataset + '_train.pt in pytorch format!')

    # Train, val, and test datasets
    train_data = TestbedDataset(
        root=root,
        dataset='train_' + dataset,
        xd=xd_train,
        xt=xc_train,
        y=y_train,
        smile_graph=smile_graph)
    val_data = TestbedDataset(
        root=root,
        dataset='val_' + dataset,
        xd=xd_val,
        xt=xc_val,
        y=y_val,
        smile_graph=smile_graph)
    test_data = TestbedDataset(
        root=root,
        dataset='test_' + dataset,
        xd=xd_test,
        xt=xc_test,
        y=y_test,
        smile_graph=smile_graph)

    # All samples dataset
    all_data = TestbedDataset(
        root=root,
        dataset='all_' + dataset,
        xd=xd_all,
        xt=xc_all,
        y=y_all,
        smile_graph=smile_graph)


"""
The functions below generate datasets for CSG (data from July 2020) - End
"""


if __name__ == "__main__":
    fdir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument(
        '--choice',
        type=int,
        required=False,
        default=0,
        help='0. mix test, 1. saliency value, 2. drug blind, 3. cell blind, 4. csg_data, 5. raw_drp_to_ml_data')
    parser.add_argument(
        '--outdir',
        type=str,
        required=False,
        default="data_processed",
        help='Data dir name to store the preprocessed data.')
    # -------------------
    # That's for CSG analysis
    parser.add_argument(
        '--split',
        type=int,
        required=False,
        default=0,
        help='Split id in the cross-stugy analysis.')
    # That's for CSG analysis
    parser.add_argument(
        '--split_file_path',
        type=str,
        required=False,
        help='Split file path in the cross-stugy analysis.')
    # -------------------

    args = parser.parse_args()

    choice = args.choice

    if choice == 4:
        # CSG (from July 2020)
        ftp_origin = "https://ftp.mcs.anl.gov/pub/candle/public/improve/cross_study_gen/"
        fname = "csg_data.zip"
        datadir = fdir

        csg_datadir = candle.get_file(
            fname=fname,
            origin=os.path.join(ftp_origin, fname),
            unpack=True, md5_hash=None,
            datadir=datadir,
            cache_subdir=None)
    else:
        # Original data
        ftp_origin = "https://ftp.mcs.anl.gov/pub/candle/public/improve/reproducability/GraphDRP/data"
        datadir = fdir/"data"

        ftp_fname = fdir/"ftp_file_list"
        with open(ftp_fname, "r") as f:
            data_file_list = f.readlines()

        for f in data_file_list:
            candle.get_file(
                fname=f.strip(),
                origin=os.path.join(ftp_origin, f.strip()),
                unpack=False, md5_hash=None,
                datadir=datadir,
                cache_subdir=None)

    if choice == 5:
        # TODO: use get_file() instead of hard-coding it
        # TODO: consider to replicate choice == 4
        raw_drp_datadir = fdir/"ml.dfs.tr_vl_te/July2020/data.gdsc2"

    if choice == 0:
        # save mix test dataset
        save_mix_drug_cell_matrix(args)
    elif choice == 1:
        # save saliency map dataset
        save_best_individual_drug_cell_matrix(args)
    elif choice == 2:
        # save blind drug dataset
        save_blind_drug_matrix(args)
    elif choice == 3:
        # save blind cell dataset
        save_blind_cell_matrix(args)
    elif choice == 4:
        # CSG data
        gen_csg_data(args, csg_datadir)
    elif choice == 5:
        # CSG data
        raw_drp_to_ml_data(args, raw_drp_datadir)
    else:
        print("Invalid option, choose 0 -> 5")

    print("Finished pre-processing.")
