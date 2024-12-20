import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import pickle
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch.nn as nn
import argparse

import wandb
from dotenv import load_dotenv


class GraphNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, mlp_hidden_channels=256, num_classes=1):
        super(GraphNet, self).__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_channels, mlp_hidden_channels),
            nn.ReLU(),
            nn.Linear(mlp_hidden_channels, num_classes)
        )

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        edge_prediction = self.mlp(edge_features)

        return edge_prediction.view(-1)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--gpu", default=1, type=int, help="GPU id to use. Default is 1.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu"], type=str, help="cpu or gpu")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPUs ID")

    parser.add_argument(
        "--split",
        default="RandomTCR",
        type=str,
        choices=["RandomTCR", "StrictTCR"],
        help="Choose split method: RandomTCR or StrictTCR."
    )
    parser.add_argument(
        "--dataset",
        default="pMTnet",
        type=str,
        choices=["McPAS", "pMTnet", "VDJdb", "TEINet"],
        help="Choose from McPAS, pMTnet, VDJdb, TEINet."
    )
    
    return parser.parse_args()


def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)


def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)


def get_test_data(test_path,embedding_path):

    with open(embedding_path, 'rb') as f:
        embedding_dict = pickle.load(f)

    node_index = {} 
    num_nodes = 0
    edge_list = []
    X = []
    y_list = []
    data = pd.read_csv(test_path)
    for _, row in data.iterrows():
        label = float(row["Label"])
        nodes = [row["Epitope"], row["CDR3.beta"]]
        for node in nodes:
            if node not in node_index:
                node_index[node] = num_nodes
                num_nodes += 1
                X.append(embedding_dict[node])
        y_list.append(label)
        edge_list.append((node_index[nodes[0]], node_index[nodes[1]]))

    
    X = torch.tensor(np.array(X), dtype=torch.float)
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    y = torch.tensor(y_list, dtype=torch.float)


    return Data(x=X, edge_index=edge_index, y=y, num_nodes=num_nodes)

#   python inference.py --split RandomTCR --dataset pMTnet --device gpu --gpu_id 0

# -----------------------------------------------------------------------------

# hyperparameter_tuning_with_WnB = False

MODEL_NAME = "GNN"

# -----------------------------------------------------------------------------
# W&B Setup
# -----------------------------------------------------------------------------

experiment_name = f"Experiment - {MODEL_NAME}"
load_dotenv()
# PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
PROJECT_NAME = f"dataset-inference_GNN" 
print(f"PROJECT_NAME: {PROJECT_NAME}")

# Initialize WandB
wandb.init(
    project=PROJECT_NAME,
    name=experiment_name,
    entity="pa_cancerimmunotherapy",
    config={
        "model_name": MODEL_NAME,
        
        # "device": str(device),
        "precision_levels": ["allele", "gene"]
        # "train_folds": train_folds,
        # "embedding_path": embedding_path,
    }
)


# ------------------------------------------------

args = parse_args()
# device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

if args.device == "cpu" or not torch.cuda.is_available():
    device = torch.device("cpu")
elif args.device == "gpu":
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

# dataset = args.dataset
# split = args.split

# print(f"You chose the dataset: {dataset}")
# print(f"The split method is: {split}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("torch.cuda.is_available: ", torch.cuda.is_available())


for i in range(4,5):  # this is just i = 4, once.
    train_folds = ''.join([str(j) for j in range(5) if j != i])  # this is just '0123'

# =======================================
# prepare test.csv from BA to be infered in this script

precisions = ["allele", "gene"]


for precision in precisions:

    # data preparation
    path_to_test = f"./data_for_inference/{precision}/beta/test.tsv"
    print("Processing file: ", path_to_test)
    df = pd.read_csv(path_to_test, sep='\t')
    df = df[['TRB_CDR3', 'Epitope', 'Binding']] # keep just what is needed

    columns_to_rename = {
        'TRB_CDR3': 'CDR3.beta',
        'Binding': 'Label'
    }
    df.rename(columns=columns_to_rename, inplace=True)
    # print('Satarting length= ', len(df))
    df = df[df['CDR3.beta'].apply(len) <= 30] 
    # print('Length after removing CDR3.beta len > 30 = ', len(df))
    df = df[df['Epitope'].apply(len) <= 30] 
    # print('Length after removing Epitope len > 30 = ', len(df))
    # save file
    save_path = f"processed_data/PA_all/{precision}_test.csv"
    df.to_csv(save_path, index=False)

# ======================================
    
    file_path = f"processed_data/PA_all/{precision}_test.csv"
    
    # model_path = f"models/{dataset}/{split}/{dataset}_{train_folds}_{i}.pth"
    model_path = f"./results/PA_all_{precision}_{train_folds}_{i}.pth"
    
    embedding_path = "./models/PA_all/gene_and_allele_embeddings.pkl"
    
    test_data = get_test_data(file_path, embedding_path).to(device)
    test_data_df = pd.read_csv(file_path)
    
    GTE = GraphNet(num_node_features=test_data.num_node_features).to(device)
    GTE.load_state_dict(torch.load(model_path, map_location=device))
    GTE.eval()
    
    with torch.no_grad():
        preds_test = GTE(test_data.x, test_data.edge_index)
        y_true_test = test_data.y.to(device)
    
        roc_auc_test = compute_auc(preds_test, y_true_test)
        test_aupr = compute_aupr(preds_test, y_true_test)
    
        # save results
        probabilities = torch.sigmoid(preds_test)
        binary_predictions = (probabilities > 0.5).type(torch.int).detach().cpu().numpy()
        df = pd.DataFrame({
            "CDR3.beta":test_data_df["CDR3.beta"].values,
            "Epitope":test_data_df["Epitope"].values,
            'Label': y_true_test.detach().cpu().numpy().astype(int),
            'Prediction': probabilities.detach().cpu().numpy(),
            
        })
        df.to_csv(f'results/inference_{precision}_test.csv', index=False)
    
        print(f"Fold: {i}, AUC: {roc_auc_test:.4f}, AUPR: {test_aupr:.4f}")
        
                # Log Metrics and Results to WandB
        wandb.log({
            "precision": precision,
            # "fold": i,
            "roc_auc_test": roc_auc_test,
            "aupr_test": test_aupr,
            "num_test_samples": len(test_data),
        })

# Finalize WandB Run
wandb.finish()
