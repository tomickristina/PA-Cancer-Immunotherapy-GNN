import torch
import torch.nn.functional as F
from data_processing import TEINet_embeddings_tvt
from model import GraphNet
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import pandas as pd
from libauc.losses import AUCMLoss
from libauc.optimizers import PESG
from arg_parser import parse_args
import numpy as np
import collections
from torch_geometric.data import Data
import random
from sklearn.model_selection import train_test_split
import yaml

from dotenv import load_dotenv
import wandb
import os



seed = 18
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def compute_accuracy(preds, y_true):
    return ((preds > 0).float() == y_true).sum().item() / preds.size(0)


def compute_aupr(preds, y_true):
    probs = torch.sigmoid(preds)
    probs_numpy = probs.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)

def compute_ap(preds, y_true):
    # probs = torch.sigmoid(preds)
    # probs_numpy = probs.detach().cpu().numpy()
    probs_numpy = preds.detach().cpu().numpy()
    y_true_numpy = y_true.detach().cpu().numpy()
    return average_precision_score(y_true_numpy, probs_numpy)

def compute_auc(preds, y_true):
    probs = torch.sigmoid(preds)
    y_true_numpy = y_true.detach().cpu().numpy()
    probs_numpy = probs.detach().cpu().numpy()
    return roc_auc_score(y_true_numpy, probs_numpy)



args = parse_args()
print(args)
with open(args.configs_path) as file:
    configs = yaml.safe_load(file)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------

# precision = "gene" # or allele
precision = "gene"         

hyperparameter_tuning_with_WnB = False

MODEL_NAME = "GNN"

# -----------------------------------------------------------------------------
# W&B Setup
# -----------------------------------------------------------------------------

experiment_name = f"Experiment - {MODEL_NAME}"
load_dotenv()
# PROJECT_NAME = os.getenv("MAIN_PROJECT_NAME")
PROJECT_NAME = f"dataset-all-tvt-{precision}_GNN" # or allele 
print(f"PROJECT_NAME: {PROJECT_NAME}")
run = wandb.init(project=PROJECT_NAME, job_type=f"{experiment_name}", entity="pa_cancerimmunotherapy", config={     
    "num_epochs": args.epochs,
    "learning_rate": args.lr,
    "positive_weights": args.positive_weights,
    "w_celoss": args.w_celoss,
    "dataset": configs["dataset_name"]})


# -----------------------------------------------------------------------------
# data (from W&B)
# -----------------------------------------------------------------------------
# Download corresponding artifact (= dataset) from W&B
dataset_names = ['train.csv', 'validation.csv', 'test.csv']
for dataset_name in dataset_names:
    artifact = run.use_artifact(f"{dataset_name}:latest")
    data_dir = artifact.download(f"./WnB_Experiments_Datasets/{dataset_name}")

# ------------------------------------------------------------------------------

data_list = TEINet_embeddings_tvt(args.configs_path)

data_list = [data.to(device) for data in data_list]

train_data = data_list[0]
validation_data = data_list[1]
test_data = data_list[2]


model = GraphNet(num_node_features=train_data.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)  # try another optimizer

# ---------------------------------------------------------------------------------------
if hyperparameter_tuning_with_WnB:
    hyperparameters = set_hyperparameters(config)  # hyperparameter tuning with Weight&Biases sweeps
# ---------------------------------------------------------------------------------------

margin = 4.0
epoch_decay = 0.0046
weight_decay = 0.006
aucm_optimizer = PESG(model.parameters(),
                 loss_fn=AUCMLoss(),
                 lr=args.lr,
                 momentum=0.4,
                 margin=margin,
                 device=device,
                 epoch_decay=epoch_decay,
                 weight_decay=weight_decay)


num_epochs = args.epochs
best_valid_roc = 0
best_valid_acc = 0

# ---------------------------------------------------------------------------------

# This logs gradients
wandb.watch(model, log="all", log_freq=10)

print("wandb logger initialisiert")
# Callbacks
run_name = wandb.run.name  

# ---------------------------------------------------------------------------------------


for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    aucm_optimizer.zero_grad()

    out = model(train_data.x, train_data.edge_index)
    preds = out
    y_true = train_data.y.to(device)

    num_positive_samples = (y_true == 1).sum()
    num_negative_samples = (y_true == 0).sum()
    weight_factor = num_negative_samples.float() / num_positive_samples.float()
    pos_weight = torch.ones([y_true.size(0)],device=device) * weight_factor * args.positive_weights
    bce_loss = F.binary_cross_entropy_with_logits(preds, y_true, pos_weight=pos_weight)


    aucm_module = AUCMLoss()
    aucm_loss = aucm_module(torch.sigmoid(preds), y_true)
    total_loss = args.w_celoss * bce_loss + args.w_aucloss * aucm_loss.to(device)
    total_loss.backward()
    optimizer.step()
    aucm_optimizer.step()

    accuracy = compute_accuracy(preds, y_true)
    roc_auc = compute_auc(preds, y_true)
    aupr = compute_aupr(preds, y_true)
    # ap = average_precision_score(y_true, preds)  # AP computation

    wandb.log({
        "epoch": epoch + 1,
        "train_loss": total_loss.item(),
        "train_accuracy": accuracy,
        "train_roc_auc": roc_auc,
        "train_ap": aupr
        # "train_ap": ap
    })
    # Validation part
    model.eval()
    with torch.no_grad():
        out_valid = model(validation_data.x, validation_data.edge_index)
        preds_valid = out_valid
        y_true_valid = validation_data.y.to(device)


        valid_acc = compute_accuracy(preds_valid, y_true_valid)
        roc_auc_valid = compute_auc(preds_valid, y_true_valid)
        valid_aupr = compute_aupr(preds_valid, y_true_valid)
        # valid_ap = average_precision_score(y_true_valid, preds_valid)  # AP computation

        if roc_auc_valid > best_valid_roc:
            best_valid_roc = roc_auc_valid
            torch.save(model.state_dict(), configs['save_model'])


        # Log validation metrics to W&B
        wandb.log({
            "epoch": epoch + 1,
            "valid_accuracy": valid_acc,
            "valid_roc_auc": roc_auc_valid,
            "valid_ap": valid_aupr
            # "valid_ap": valid_ap  # Log Average Precision
        })

    
    print("Epoch: {}/{}, Loss: {:.7f}, Train Acc: {:.4f}, Valid. Acc: {:.4f}, Train AUC: {:.4f}, Train AP: {:.4f}, Valid. AUC: {:.4f}, Valid. AP: {:.4f}".format(epoch+1, num_epochs, total_loss.item(), accuracy, valid_acc, roc_auc, aupr, roc_auc_valid, valid_aupr))
    
# Load the best model
best_model = GraphNet(num_node_features=test_data.num_node_features).to(device)
best_model.load_state_dict(torch.load(configs['save_model']))

# Log model artifact to W&B
wandb.log_artifact(configs['save_model'], type="model", name="best_model")

# Evaluate on test test_data
best_model.eval()
with torch.no_grad():
    out_test = best_model(test_data.x, test_data.edge_index)
    preds_test = out_test
    y_true_test = test_data.y.to(device)

    test_acc = compute_accuracy(preds_test, y_true_test)
    roc_auc_test = compute_auc(preds_test, y_true_test)
    test_aupr = compute_aupr(preds_test, y_true_test)
    # test_ap = average_precision_score(y_true_test, preds_test)  # AP computation
# ------------------------
    # Convert logits to binary predictions
    probabilities_test = torch.sigmoid(preds_test)
    binary_predictions_test = (probabilities_test > 0.5).type(torch.int).detach().cpu().numpy()
    labels_test = y_true_test.detach().cpu().numpy()

    # Compute confusion matrix
    conf_matrix_test = confusion_matrix(labels_test, binary_predictions_test)

    # Log test results and confusion matrix to W&B
    wandb.log({
        "test_accuracy": test_acc,
        "test_roc_auc": roc_auc_test,
        "test_ap": test_aupr,
        # "test_ap": test_ap,  # Log Average Precision
        "test_confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels_test,
            preds=binary_predictions_test,
            class_names=["Not Binding", "Binding"]
        )
    })

    

    
# ---------------------------------------
    df = pd.DataFrame({
        'prediction': binary_predictions_test,
        'label': y_true_test.detach().cpu().numpy().astype(int)
    })

# ======================================================================    
    path_test = f"./processed_data/PA_all_tvt/{precision}_test.csv"
    # or 
    # path_test = "./processed_data/PA_all_tvt/gene_test.csv"
    dfr = pd.read_csv(path_test)
    
    df['task'] = dfr['task']

    tasks = df['task'].unique()

    # Calculate AP and ROC AUC for each task
    print("Task-wise AP and ROC AUC:")
    for task in tasks:
        # Filter the DataFrame for the current task
        df_task = df[df['task'] == task]
        
        # Check for at least two classes in the labels
        if len(df_task['label'].unique()) > 1:
            # Use probabilities instead of binary predictions
            task_probs = probabilities_test[df['task'] == task].detach().cpu().numpy()  # Probabilities for this task
            task_labels = df_task['label'].values               # True labels for this task

            ap = average_precision_score(task_labels, task_probs)
            roc_auc = roc_auc_score(task_labels, task_probs)
            print(f"Task: {task}, AP: {ap:.4f}, ROC AUC: {roc_auc:.4f}")
        else:
            print(f"Task: {task}, AP and ROC AUC cannot be calculated (only one class present).")

    
    # Calculate AP and ROC AUC for the entire dataset
    print("\nOverall AP and ROC AUC:")
    if len(df['label'].unique()) > 1:
        # Convert probabilities_test to a NumPy array
        overall_probs = probabilities_test.detach().cpu().numpy()  # Move to CPU and convert to NumPy
        overall_labels = df['label'].values                        # True labels as a NumPy array
    
        overall_ap = average_precision_score(overall_labels, overall_probs)
        overall_roc_auc = roc_auc_score(overall_labels, overall_probs)
    
        print(f"Overall AP: {overall_ap:.4f}, Overall ROC AUC: {overall_roc_auc:.4f}")
    else:
        print("AP and ROC AUC cannot be calculated for the overall dataset (only one class present).")

    
    # # Calculate AP and ROC AUC for the entire dataset
    # print("\nOverall AP and ROC AUC:")
    # if len(df['label'].unique()) > 1:
    #     overall_ap = average_precision_score(df['label'], probabilities_test)
    #     overall_roc_auc = roc_auc_score(df['label'], probabilities_test)

    #     # overall_ap = average_precision_score(df['label'], df['prediction'])
    #     # overall_roc_auc = roc_auc_score(df['label'], df['prediction'])
    #     print(f"Overall AP: {overall_ap:.4f}, Overall ROC AUC: {overall_roc_auc:.4f}")
    # else:
    #     print("AP and ROC AUC cannot be calculated for the overall dataset (only one class present).")

# =============================================================================
    
    
    results_file = f'results/{configs["dataset_name"]}.csv'
    df.to_csv(f'results/{configs["dataset_name"]}.csv', index=False)

    # Log predictions file to W&B
    wandb.log_artifact(results_file, type="results", name="test_predictions")

    



wandb.finish()


print("Test Acc: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}".format(test_acc, roc_auc_test, test_aupr))