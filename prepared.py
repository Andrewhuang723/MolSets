## Generates the Molecular graph pickle files
import pdb
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from data_utils import RevIndexedDataset
from dmpnn import MolSets_DMPNN
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from data_feature import smiles2data, SMILES_COMPOUND, solvent_cols
from scipy.optimize import differential_evolution, NonlinearConstraint, LinearConstraint

## Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


## Standardization from "training dataset"
dataset = RevIndexedDataset('./data/data_list_v2.pkl')
target_mean = dataset.target_mean
target_std = dataset.target_std

checkpoint = torch.load("./results/DMPNN_6_h1024_e1024_reg0.0001_norm.pt")
best_model = checkpoint["best_model"]
hyperpars = checkpoint["hyperparameters"]
model = MolSets_DMPNN(n_node_features=133, hidden_dim=hyperpars['hidden_dim'], emb_dim=hyperpars['emb_dim'], 
                    output_dim=1, n_conv_layers=hyperpars['n_conv_layers'], after_readout=hyperpars['after_readout']).to(device)

model.load_state_dict(best_model)


def predict(test_data: RevIndexedDataset, model: MolSets_DMPNN) -> list:
    results = np.zeros(len(test_data))
    model.eval()
    with torch.no_grad():
        for i in range(len(test_data)):
            index, inputs, mw, frac, salt_mol, salt_graph, target = test_data.get(i)
            inputs = Batch.from_data_list(inputs).to(device)
            target = torch.tensor(target).to(device)
            frac = torch.tensor(frac).to(device)
            salt_mol = torch.tensor(salt_mol).to(device)
            mw = torch.tensor(mw).to(device)
            salt_graph.to(device)
            out = model(inputs, mw, frac, salt_mol, salt_graph)
            out = out.cpu().numpy()
            results[i] += out
    return results

## Create Mixtures

MIXTURE = lambda EC, PC, EMC, DMC, DEC, FEC, salt, c: {
    "EC": EC,
    "PC": PC,
    "EMC":EMC,
    "DMC": DMC,
    "DEC": DEC,
    "FEC": FEC,
    "salt": salt,
    "c": c
}

def create_mixtures(mixtures: list[dict]):
    mixture_list = []
    for i in range(len(mixtures)):
        ID = i + 1 # int
        graphs = [] # list[torch_geometric.data]
        mw = [] # list
        fraction = [] # list
        salt = mixtures[i]["salt"]
        salt_mol = mixtures[i]["c"] # float
        
        salt_smiles = SMILES_COMPOUND[salt]
        salt_features = smiles2data(salt_smiles)
        

        for solvent in solvent_cols:
            if mixtures[i][solvent] > 0:
                solvent_smiles = SMILES_COMPOUND[solvent]
                solvent_mol = Chem.MolFromSmiles(solvent_smiles)
                graphs.append(smiles2data(solvent_smiles))
                fraction.append(mixtures[i][solvent])
                mw.append(rdMolDescriptors.CalcExactMolWt(solvent_mol))
        
        recipe = (ID, graphs, mw, fraction, salt_mol, salt_features, [0])
        mixture_list.append(recipe)
    return mixture_list


def objective_function(x, salt="LiPF6"):
        xs = [float(x[i]) for i in range(len(x))]
        xs.insert(-1, salt)
        mixture = [MIXTURE(*xs)]  # A simple 2D function (minimizes at [0, 0])
        samples = create_mixtures(mixtures=mixture)
        with open("./data/temp_samples.pkl", "wb") as f:
            pickle.dump(samples, f)
        test_data = RevIndexedDataset('./data/temp_samples.pkl', mean=target_mean, std=target_std)

        out = predict(test_data=test_data, model=model)
        out = np.power(10, dataset.get_orig(out)) * 1000
        return out * -1

def constraint_eq(x):
    return sum(x[:6]) - 1

def iteration_callback(xk, convergence):
    print(f"Iteration: Current best solution {xk}, Convergence: {convergence}")


if __name__ == "__main__":

    # Define the objective function to minimize

    # Define bounds for the variables
    bounds = [(1e-3, 1.00), (1e-3, 0.10), (1e-3, 1.00), (0.35, 0.35), (0.00, 0.00), (0.00, 0.00), (1.50, 1.50)]
    nlc_eq = NonlinearConstraint(constraint_eq, 0, 0)  

    A = [[1, 1, 1, 1, 1, 1, 0]]
    lower_bound = 1.0
    upper_bound = 1.0
    lc = LinearConstraint(A=A, ub=upper_bound, lb=lower_bound)


    # Run the differential evolution algorithm
    result = differential_evolution(objective_function, bounds=bounds, constraints=lc, seed=42, callback=iteration_callback)

    # Print the result
    print('Optimal solution:', result.x)
    opt_x = result.x
    opt_x = [float(opt_x[i]) for i in range(len(opt_x))]
    opt_x.insert(-1, 'LiPF6')
    mixture = [MIXTURE(*opt_x)]  # A simple 2D function (minimizes at [0, 0])
    samples = create_mixtures(mixtures=mixture)
    with open("./data/temp_samples.pkl", "wb") as f:
        pickle.dump(samples, f)
    test_data = RevIndexedDataset('./data/temp_samples.pkl', mean=target_mean, std=target_std)

    out = predict(test_data=test_data, model=model)
    out = np.power(10, dataset.get_orig(out)) * 1000

    print('Objective function value at the optimum:', out)