from typing import List, Union
import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


def mol2data(mol):
    atom_feat = [atom_features(atom) for atom in mol.GetAtoms()]

    edge_attr = []
    edge_index = []

    for bond in mol.GetBonds():
        # eid = bond.GetIdx()
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.extend([(i, j), (j, i)])
        b = bond_features(bond)
        edge_attr.extend([b, b.copy()])

    x = torch.FloatTensor(atom_feat)
    edge_attr = torch.FloatTensor(edge_attr)
    edge_index = torch.LongTensor(edge_index).T

    graph = Data(x=x, edge_attr=edge_attr, edge_index=edge_index)
    graph.num_nodes = len(atom_feat)

    return graph


def smiles2data(smi, explicit_h=True):
    mol = Chem.MolFromSmiles(smi)
    if explicit_h:
        mol = Chem.AddHs(mol)
    return mol2data(mol)

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    "atomic_num": list(range(MAX_ATOMIC_NUM)),
    "degree": [0, 1, 2, 3, 4, 5],
    "formal_charge": [-1, -2, 1, 2, 0],
    "chiral_tag": [0, 1, 2, 3],
    "num_Hs": [0, 1, 2, 3, 4],
    "hybridization": [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ],
}

BOND_FDIM = 14


def onek_encoding_unk(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(atom):
    features = (
        onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES["num_Hs"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), ATOM_FEATURES["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
        + [atom.GetMass() * 0.01]
    )  # scaled to about the same range as other features
    return features


def bond_features(bond):
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond

SMILES_COMPOUND = {
    "EC": "O=C1OCCO1",
    "PC": "CC1COC(=O)O1",
    "EMC": "CCOC(OC)=O",
    "DMC": "COC(=O)OC",
    "DEC": "CCOC(=O)OCC",
    "FEC": "C1C(OC(=O)O1)F",
    "LiPF6": "F[P-](F)(F)(F)(F)F.[Li+]",
    "LiBOB": "[Li+].[B-]12(OC(=O)C(=O)O1)OC(=O)C(=O)O2",
    "LiBF4": "[Li+].[B-](F)(F)(F)F",
}


solvent_cols = ["EC", "PC", "EMC", "DMC", "DEC", "FEC"]



if __name__ == "__main__":
    # print(smiles2data("c1ccccc1").to_dict())
    import pandas as pd

    df = pd.read_csv("./data/CALiSOL_v1.csv")
    df1_length = 16009
    feature_list = []


    for i in range(len(df)):
        ID = df1_length + i + 1 # int
        graphs = [] # list[torch_geometric.data]
        mw = [] # list
        fraction = [] # list
        salt_mol = float(df.loc[i]["c"]) # float
        
        salt_smiles = SMILES_COMPOUND[df.loc[i]["salt"]]
        salt_features = smiles2data(salt_smiles)
        
        target = [np.log10(df.loc[i]["k"] / 1000)]

        for solvent in solvent_cols:
            if df.loc[i][solvent] > 0:
                solvent_smiles = SMILES_COMPOUND[solvent]
                solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                graphs.append(smiles2data(solvent_smiles))
                fraction.append(float(df.loc[i][solvent]))
                mw.append(rdMolDescriptors.CalcExactMolWt(solvent_mol))
        
        recipe = (ID, graphs, mw, fraction, salt_mol, salt_features, target)
        feature_list.append(recipe)

    feature_data_df = pd.DataFrame(feature_list)
    feature_data_df.columns = ["index", 'graphs', 'mw', 'fraction', 'salt_mol', 'salt', 'target']
    feature_data_df["mol_type"] = ["small"] * len(feature_data_df)
    feature_data_df["comp_type"] = 1000
    feature_data_df.set_index("index", inplace=True)

    import pickle

    with open('./data/data_list_v1.pkl', 'wb') as f:
        pickle.dump(feature_list, f)

    with open('./data/data_df_stats_v1.pkl', 'wb') as f:
        pickle.dump(feature_data_df, f)