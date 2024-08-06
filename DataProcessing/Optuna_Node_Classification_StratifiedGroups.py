import json
import logging

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphConv
from torch_geometric.nn import global_mean_pool, BatchNorm, GraphNorm, GATv2Conv


class GNN1(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int,
                 normalise_graph: bool = False, gnn_task: str = "node_classification"):
        super().__init__()
        torch.manual_seed(12345)
        self.normalise = normalise_graph
        # affine=False only does a normalisation for this batch and not learn over everything
        # affine=True will learn over the full training data
        # track_running_stats default True so that the eval() uses the values computed in the training
        self.batch_norm = BatchNorm(in_channels=num_node_features, affine=True)
        self.graph_norm = GraphNorm(in_channels=num_node_features)  # https://arxiv.org/pdf/2009.03294.pdf

        self.task = gnn_task
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)  # GraphConv
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch: int = None):
        if self.task == "graph_classification" and self.normalise:
            # 1. Normalise Node Features on a batch-based normalisation for a graph classification problem
            # Source: https://arxiv.org/abs/2009.11746
            x = self.batch_norm(x)
        elif self.task == "node_classification" and self.normalise:
            # 1. Normalise Node Features on a graph-based normalisation for a node classification problem
            # Source: https://arxiv.org/abs/2009.11746
            x = self.graph_norm(x)
            # print("Normalised")

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 3. Readout layer
        if self.task == "graph_classification":
            # Returns batch-wise graph-level-outputs by averaging node features across the node dimension
            x = global_mean_pool(x, batch)  # [batch_size, hidden_layer_size]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GNN2(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int,
                 normalise_graph: bool = False, gnn_task: str = "graph_classification"):
        super().__init__()
        torch.manual_seed(12345)
        self.normalise = normalise_graph
        # affine=False only does a normalisation for this batch and not learn over everything
        # affine=True will learn over the full training data
        # track_running_stats default True so that the eval() uses the values computed in the training
        self.batch_norm = BatchNorm(in_channels=num_node_features, affine=True)
        self.graph_norm = GraphNorm(in_channels=num_node_features)  # https://arxiv.org/pdf/2009.03294.pdf

        self.task = gnn_task
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch: int = None):
        if self.task == "graph_classification" and self.normalise:
            # 1. Normalise Node Features on a batch-based normalisation for a graph classification problem
            # Source: https://arxiv.org/abs/2009.11746
            x = self.batch_norm(x)
        elif self.task == "node_classification" and self.normalise:
            # 1. Normalise Node Features on a graph-based normalisation for a node classification problem
            # Source: https://arxiv.org/abs/2009.11746
            x = self.graph_norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 3. Readout layer
        if self.task == "graph_classification":
            # Returns batch-wise graph-level-outputs by averaging node features across the node dimension
            x = global_mean_pool(x, batch)  # [batch_size, hidden_layer_size]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class GNN3(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_node_features: int, num_classes: int, num_heads: int,
                 normalise_graph: bool = True, gnn_task: str = "node_classification"):
        super().__init__()
        torch.manual_seed(12345)
        self.task = gnn_task
        self.normalise = normalise_graph
        # affine=False only does a normalisation for this batch and not learn over everything
        # affine=True will learn over the full training data
        # track_running_stats default True so that the eval() uses the values computed in the training
        self.batch_norm = BatchNorm(in_channels=num_node_features, affine=True)
        self.graph_norm = GraphNorm(in_channels=num_node_features)  # https://arxiv.org/pdf/2009.03294.pdf

        self.task = gnn_task

        # The number of heads is a parameter to optimise, and will try 1, 2, 4, 8 as reported in the original paper
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, heads=num_heads)  # GATConv
        self.conv2 = GATv2Conv(num_heads * hidden_channels, hidden_channels, heads=num_heads)
        self.conv3 = GATv2Conv(num_heads * hidden_channels, hidden_channels, heads=1, concat=False)
        self.lin = Linear(hidden_channels, num_classes)
        print("GAT Initialised")

    def forward(self, x, edge_index, batch: int = None):
        if self.task == "graph_classification" and self.normalise:
            # 1. Normalise Node Features on a batch-based normalisation for a graph classification problem
            # Source: https://arxiv.org/abs/2009.11746
            x = self.batch_norm(x)
        elif self.task == "node_classification" and self.normalise:
            # 1. Normalise Node Features on a graph-based normalisation for a node classification problem
            # Source: https://arxiv.org/abs/2009.11746
            x = self.graph_norm(x)

        # 2. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = self.conv3(x, edge_index)
        # 3. Readout layer
        if self.task == "graph_classification":
            # Returns batch-wise graph-level-outputs by averaging node features across the node dimension
            x = global_mean_pool(x, batch)  # [batch_size, hidden_layer_size]

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


class Processing:
    def __init__(self, corpus: str, aoi_type: str, num_node_features: int, which_data_type: str,
                 which_model: str, number_of_trials: int, gpu: int, number_of_jobs: int):
        super().__init__()
        self.Corpus = corpus
        self.AOI_Type = aoi_type
        self.Number_Of_Node_Features_File_Name = num_node_features
        self.which_data_type = which_data_type
        self.Which_Model = which_model
        self.number_of_trials = number_of_trials
        self.number_of_jobs = number_of_jobs
        self.scanpath_df_all, self.scanpath_df_agree = self.load_data()
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        if self.Corpus == "gREL":
            self.Graph_Task = "graph_classification"
        else:
            self.Graph_Task = "node_classification"
        self.num_node_features = None
        self.list_full_training_data = None
        self.list_full_training_labels = None
        self.list_full_training_user_ids = None
        print(f"Device is: {self.device}")
        self.epoch = [100, 200, 300, 400, 500]
        self.hidden_size = [8, 16, 32, 64, 128, 256]
        self.learning_rate = [0.1, 0.01, 0.001, 0.0001]
        self.batch_size = [8, 16, 32, 64]
        self.normalise = [True, False]

    def prepare_data(self, df: pd.DataFrame):
        """
        Formats the dataframe into a proper graph format for the GNNs.
        The dataframe must have the following columns:
            - edge_index
            - node_features
            - label
            - user_id for the leave-one-user-out cross-validation
        """
        dataset_list = []
        labels_for_split_list = []
        user_groups_list = []
        num_node_features = 0
        for ind in df.index:
            """
            The df was saved as a CSV file, which saves lists as strings. 
            We use json.loads to extract it as a list of lists once again.
            """
            edges = json.loads(df.loc[ind, "edge_index"])
            node_features = json.loads(df.loc[ind, "node_features"])
            label = json.loads(df.loc[ind, "label"])
            num_node_features = len(node_features[0])
            edge_index = torch.tensor(edges, dtype=torch.long)
            x = torch.tensor(node_features, dtype=torch.float)
            y = torch.tensor(label, dtype=torch.int64)
            data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
            dataset_list.append(data)
            labels_for_split_list.append(len(label))
            user_id = df.loc[ind, "user_id"]
            user_groups_list.append(user_id)

        return dataset_list, labels_for_split_list, user_groups_list, num_node_features

    def evaluate_results(self, true_labels, prediction_labels, prediction_proba):
        accuracy = balanced_accuracy_score(true_labels, prediction_labels)
        loss_score = log_loss(y_true=true_labels, y_pred=prediction_labels, labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix(y_true=true_labels, y_pred=prediction_labels, labels=[0, 1]).ravel()
        if (tp + fp) == 0 and (tp + fn) == 0:
            print("F1 Zero")
            f1_result = 0
        else:
            f1_result = f1_score(true_labels, prediction_labels, labels=[0, 1])
        if tp + fn == 0:
            print("TPR Zero")
            tpr = 0
        else:
            tpr = tp / (tp + fn)
        if tn + fp == 0:
            print("FPR Zero")
            fpr = 0
        else:
            fpr = fp / (tn + fp)
        try:
            roc_auc = roc_auc_score(y_true=true_labels, y_score=prediction_proba, average="weighted", labels=[0, 1])
        except ValueError:
            print("Exception: Only one class in the true labels")
            roc_auc = 0
        return f1_result, accuracy, tpr, fpr, loss_score, roc_auc, tn, fp, fn, tp

    def load_data(self):
        """
        Corpus
            - GoogleNQ
        AOI_Type
            - Paragraph_Node_GNN
        Num_Node_Features
            - 2
            - 17
        """
        scanpath_all = pd.read_csv(
            f"../Data/Graph_Data/GoogleNQ_{self.AOI_Type}_{self.Number_Of_Node_Features_File_Name}_Features.csv")
        scanpath_agree = scanpath_all.loc[scanpath_all["label"] == scanpath_all["system_label"]]
        return scanpath_all, scanpath_agree

    def train_validate_model(self, list_dataset_training, list_dataset_val, epoch, hidden_size, learning_rate,
                             batch_size, normalise, num_gat_heads=1):
        # Load Data for GNN
        dataloader_training = DataLoader(list_dataset_training, batch_size=batch_size,
                                         shuffle=True)
        dataloader_val = DataLoader(list_dataset_val, batch_size=batch_size, shuffle=False)

        # Prepare GNN Model
        if self.Which_Model == "GNN1":
            model = GNN1(hidden_channels=hidden_size, num_node_features=self.num_node_features,
                         num_classes=2,
                         normalise_graph=normalise,
                         gnn_task=self.Graph_Task).to(self.device)
        elif self.Which_Model == "GNN2":
            model = GNN2(hidden_channels=hidden_size, num_node_features=self.num_node_features,
                         num_classes=2,
                         normalise_graph=normalise,
                         gnn_task=self.Graph_Task).to(self.device)
        else:
            model = GNN3(hidden_channels=hidden_size, num_node_features=self.num_node_features,
                         num_classes=2,
                         normalise_graph=normalise,
                         gnn_task=self.Graph_Task,
                         num_heads=num_gat_heads).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Train the neural network
        model.train().to(self.device)
        for epoch_run in range(epoch):
            # Iterate in batches over the training dataset.
            for inputs_training in dataloader_training:
                optimizer.zero_grad()  # Clear gradients.
                # Perform a single forward pass.
                outputs = model(x=inputs_training.x.to(self.device),
                                edge_index=inputs_training.edge_index.to(self.device),
                                batch=inputs_training.batch.to(self.device))
                loss = criterion(outputs,
                                 inputs_training.y.to(self.device))  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.

        # Validate the neural network
        model.eval().to(self.device)
        list_true_labels_val = []
        list_predicted_labels_val = []
        list_predicted_proba_val = []
        for inputs_val in dataloader_val:
            with torch.no_grad():  # To make sure the weights are not affected
                outputs_val = model(x=inputs_val.x.to(self.device),
                                    edge_index=inputs_val.edge_index.to(self.device),
                                    batch=inputs_val.batch.to(self.device))
                # Use the class with the highest probability.
                prediction_val = outputs_val.argmax(dim=1)
                # Get the probability for AUC
                prediction_proba_val = torch.sigmoid(outputs_val)
                list_true_labels_val.extend(inputs_val.y.tolist())
                list_predicted_labels_val.extend(prediction_val.tolist())
                list_predicted_proba_val.extend(prediction_proba_val.tolist())
        list_positive_predicted_proba_val = np.asarray(list_predicted_proba_val)[:, 1].tolist()
        # https://stackoverflow.com/questions/67753454/sklearn-roc-auc-score-valueerror-y-should-be-a-1d-array-got-an-array-of-shap

        # Evaluate the validation performance
        (f1_score_val, accuracy_val, tpr_val, fpr_val, loss_val, auc_val,
         tn_val, fp_val, fn_val, tp_val) = (
            self.evaluate_results(true_labels=list_true_labels_val,
                                  prediction_labels=list_predicted_labels_val,
                                  prediction_proba=list_positive_predicted_proba_val))
        return f1_score_val, accuracy_val, tpr_val, fpr_val, loss_val, auc_val

    def optuna_optimise_hyperparameters(self, trial):
        # Define the hyperparameters
        epoch = trial.suggest_categorical("epoch", self.epoch)
        hidden_size = trial.suggest_categorical("hidden_size", self.hidden_size)
        learning_rate = trial.suggest_categorical("learning_rate", self.learning_rate)
        batch_size = trial.suggest_categorical("batch_size", self.batch_size)
        normalise = trial.suggest_categorical("normalise", self.normalise)

        if self.Which_Model == "GNN3":
            num_gat_heads = trial.suggest_categorical("num_gat_heads", [1, 2, 4, 8])

        list_validation_accuracy = []
        list_validation_f1 = []
        list_validation_loss = []
        list_validation_tpr = []
        list_validation_fpr = []
        list_validation_auc = []

        # Pick the index of the data splits based on the groups
        inner_loop_fold_splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=256)
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

        for fold_num, (training_index, val_index) in enumerate(
                inner_loop_fold_splits.split(self.list_full_training_data, self.list_full_training_labels,
                                             self.list_full_training_user_ids)):
            list_dataset_training = [self.list_full_training_data[k] for k in training_index]
            list_dataset_val = [self.list_full_training_data[k] for k in val_index]
            if self.Which_Model == "GNN3":
                model_val_f1, model_val_accuracy, model_val_tpr, model_val_fpr, model_val_loss, model_val_auc = \
                    self.train_validate_model(
                        list_dataset_training=list_dataset_training, list_dataset_val=list_dataset_val,
                        epoch=epoch, hidden_size=hidden_size, learning_rate=learning_rate,
                        batch_size=batch_size, normalise=normalise, num_gat_heads=num_gat_heads)
            else:
                model_val_f1, model_val_accuracy, model_val_tpr, model_val_fpr, model_val_loss, model_val_auc = \
                    self.train_validate_model(
                        list_dataset_training=list_dataset_training, list_dataset_val=list_dataset_val,
                        epoch=epoch, hidden_size=hidden_size, learning_rate=learning_rate,
                        batch_size=batch_size, normalise=normalise)

            list_validation_f1.append(model_val_f1)
            list_validation_accuracy.append(model_val_accuracy)
            list_validation_tpr.append(model_val_tpr)
            list_validation_fpr.append(model_val_fpr)
            list_validation_loss.append(model_val_loss)
            list_validation_auc.append(model_val_auc)

            trial.report(model_val_accuracy, fold_num)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        average_validation_f1 = np.sum(list_validation_f1) / len(list_validation_f1)
        average_validation_accuracy = np.sum(list_validation_accuracy) / len(list_validation_accuracy)
        average_validation_tpr = np.sum(list_validation_tpr) / len(list_validation_tpr)
        average_validation_fpr = np.sum(list_validation_fpr) / len(list_validation_fpr)
        average_validation_loss = np.sum(list_validation_loss) / len(list_validation_loss)
        average_validation_auc = np.sum(list_validation_auc) / len(list_validation_auc)

        return average_validation_accuracy

    def test_train_model(self, list_dataset_testing, list_user_ids_testing):
        dict_test_results = {}
        list_user_group_ids_testing_no_duplicate = list(dict.fromkeys(list_user_ids_testing))
        string_user_group_ids_testing_no_duplicate = ', '.join(list_user_group_ids_testing_no_duplicate)

        study = optuna.create_study(direction="maximize")
        number_of_trials = self.number_of_trials
        study.optimize(self.optuna_optimise_hyperparameters, n_trials=number_of_trials, n_jobs=self.number_of_jobs)
        print("Best trial:")
        best_trial = study.best_trial
        print("\t Value: ", best_trial.value)
        print(best_trial.params)

        # Load Data for GNN
        dataloader_training = DataLoader(self.list_full_training_data, batch_size=best_trial.params["batch_size"],
                                         shuffle=True)
        dataloader_testing = DataLoader(list_dataset_testing, batch_size=best_trial.params["batch_size"],
                                        shuffle=False)
        # Initialise GNN Model
        if self.Which_Model == "GNN1":
            model = GNN1(hidden_channels=best_trial.params["hidden_size"],
                         num_node_features=self.num_node_features,
                         num_classes=2,
                         normalise_graph=best_trial.params["normalise"],
                         gnn_task=self.Graph_Task).to(self.device)
        elif self.Which_Model == "GNN2":
            model = GNN2(hidden_channels=best_trial.params["hidden_size"],
                         num_node_features=self.num_node_features,
                         num_classes=2,
                         normalise_graph=best_trial.params["normalise"],
                         gnn_task=self.Graph_Task).to(self.device)
        else:
            model = GNN3(hidden_channels=best_trial.params["hidden_size"],
                         num_node_features=self.num_node_features,
                         num_classes=2,
                         normalise_graph=best_trial.params["normalise"],
                         gnn_task=self.Graph_Task,
                         num_heads=best_trial.params["num_gat_heads"]).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_trial.params["learning_rate"])
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # Train the neural network
        model.train().to(self.device)
        list_true_labels_training = []
        list_predicted_labels_training = []
        list_predicted_proba_training = []
        for epoch_run in range(best_trial.params["epoch"]):
            # Iterate in batches over the training dataset.

            for inputs_training in dataloader_training:
                optimizer.zero_grad()  # Clear gradients.
                # Perform a single forward pass.
                outputs = model(x=inputs_training.x.to(self.device),
                                edge_index=inputs_training.edge_index.to(self.device),
                                batch=inputs_training.batch.to(self.device))
                loss = criterion(outputs, inputs_training.y.to(self.device))  # Compute the loss.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                prediction_proba_training = torch.sigmoid(outputs)
                # Save Training true and predicted labels for the last epoch
                if epoch_run == best_trial.params["epoch"] - 1:
                    list_true_labels_training.extend(inputs_training.y.tolist())
                    list_predicted_labels_training.extend(outputs.argmax(dim=1).tolist())
                    list_predicted_proba_training.extend(prediction_proba_training.tolist())
        list_positive_predicted_proba_training = np.asarray(list_predicted_proba_training)[:, 1].tolist()
        (training_f1_score, training_accuracy, training_tpr, training_fpr, training_loss, training_auc,
         training_tn, training_fp, training_fn, training_tp) = (
            self.evaluate_results(true_labels=list_true_labels_training,
                                  prediction_labels=list_predicted_labels_training,
                                  prediction_proba=list_positive_predicted_proba_training))

        # Test Model based on the best saved model from the validation loss
        model.eval().to(self.device)
        list_true_labels_testing = []
        list_predicted_labels_testing = []
        list_predicted_proba_testing = []
        for inputs_testing in dataloader_testing:
            with torch.no_grad():  # To make sure the weights are not affected
                outputs_testing = model(x=inputs_testing.x.to(self.device),
                                        edge_index=inputs_testing.edge_index.to(self.device),
                                        batch=inputs_testing.batch.to(self.device))
                prediction_testing = outputs_testing.argmax(dim=1)
                prediction_proba_testing = torch.sigmoid(outputs_testing)
                list_true_labels_testing.extend(inputs_testing.y.tolist())
                list_predicted_labels_testing.extend(prediction_testing.tolist())
                list_predicted_proba_testing.extend(prediction_proba_testing.tolist())
        list_positive_predicted_proba_testing = np.asarray(list_predicted_proba_testing)[:, 1].tolist()

        (testing_f1_score, testing_accuracy, testing_tpr, testing_fpr, testing_loss, testing_auc,
         testing_tn, testing_fp, testing_fn, testing_tp) = (
            self.evaluate_results(true_labels=list_true_labels_testing,
                                  prediction_labels=list_predicted_labels_testing,
                                  prediction_proba=list_positive_predicted_proba_testing))
        print(f"Testing Accuracy: {testing_accuracy}")
        dict_test_results["f1_score_testing"] = testing_f1_score
        dict_test_results["accuracy_testing"] = testing_accuracy
        dict_test_results["tpr_testing"] = testing_tpr
        dict_test_results["fpr_testing"] = testing_fpr
        dict_test_results["loss_testing"] = testing_loss
        dict_test_results["auc_testing"] = testing_auc
        dict_test_results["tn_testing"] = testing_tn
        dict_test_results["fp_testing"] = testing_fp
        dict_test_results["fn_testing"] = testing_fn
        dict_test_results["tp_testing"] = testing_tp
        dict_test_results["f1_score_training"] = training_f1_score
        dict_test_results["accuracy_training"] = training_accuracy
        dict_test_results["tpr_training"] = training_tpr
        dict_test_results["fpr_training"] = training_fpr
        dict_test_results["loss_training"] = training_loss
        dict_test_results["auc_training"] = training_auc
        dict_test_results["tn_training"] = training_tn
        dict_test_results["fp_training"] = training_fp
        dict_test_results["fn_training"] = training_fn
        dict_test_results["tp_training"] = training_tp
        dict_test_results["epoch"] = best_trial.params["epoch"]
        dict_test_results["hidden_size"] = best_trial.params["hidden_size"]
        dict_test_results["learning_rate"] = best_trial.params["learning_rate"]
        dict_test_results["batch_size"] = best_trial.params["batch_size"]
        dict_test_results["normalise"] = best_trial.params["normalise"]
        if self.Which_Model == "GNN3":
            dict_test_results["num_gat_heads"] = best_trial.params["num_gat_heads"]
        else:
            dict_test_results["num_gat_heads"] = 0
        dict_test_results["user_id"] = string_user_group_ids_testing_no_duplicate

        return dict_test_results

    def process_scanpaths(self, df: pd.DataFrame, data_type: str):
        # Define the hyperparameters
        df_test_results = pd.DataFrame(columns=["user_id",
                                                "epoch",
                                                "hidden_size",
                                                "learning_rate",
                                                "batch_size",
                                                "normalise",
                                                "num_gat_heads",
                                                "f1_score_testing",
                                                "accuracy_testing",
                                                "tpr_testing",
                                                "fpr_testing",
                                                "loss_testing",
                                                "auc_testing",
                                                "tn_testing",
                                                "fp_testing",
                                                "fn_testing",
                                                "tp_testing",
                                                "f1_score_training",
                                                "accuracy_training",
                                                "tpr_training",
                                                "fpr_training",
                                                "loss_training",
                                                "auc_training",
                                                "tn_training",
                                                "fp_training",
                                                "fn_training",
                                                "tp_training"
                                                ])

        # Add a row of default values
        df_test_results.loc[0] = [0] * len(df_test_results.columns)
        # Then drop this row before you start appending your actual data
        df_test_results = df_test_results.drop(0)

        list_full_data, list_full_labels, list_full_user_ids, self.num_node_features = self.prepare_data(df)

        outer_loop_fold_iterator = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=256)
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

        # Outer cross-validation loop for the testing data
        for fold_number, (training_val_index, testing_index) in enumerate(
                outer_loop_fold_iterator.split(list_full_data,
                                               list_full_labels,
                                               list_full_user_ids)):
            # Testing Data for this fold
            list_dataset_testing = [list_full_data[k] for k in testing_index]
            list_user_ids_testing = [list_full_user_ids[k] for k in testing_index]

            # Prepare data for training and validation
            self.list_full_training_data = [list_full_data[k] for k in training_val_index]
            self.list_full_training_labels = [list_full_labels[k] for k in training_val_index]
            self.list_full_training_user_ids = [list_full_user_ids[k] for k in training_val_index]

            dict_test_results = self.test_train_model(list_dataset_testing=list_dataset_testing,
                                                      list_user_ids_testing=list_user_ids_testing)

            df_test_results = pd.concat([df_test_results, pd.DataFrame([dict_test_results])], ignore_index=True)
            df_test_results.to_csv(f'../Results/GAT/'
                                   f'{self.Corpus}_{self.Which_Model}_{self.AOI_Type}_{data_type}_'
                                   f'{self.Number_Of_Node_Features_File_Name}_Features_Fold_{fold_number}.csv',
                                   index=False)

            print(f'{self.Which_Model} Fold {fold_number} Done')

    def run(self):
        print(self.which_data_type)
        if self.which_data_type == "Agree":
            self.process_scanpaths(df=self.scanpath_df_agree, data_type="agree")
        elif self.which_data_type == "All":
            self.process_scanpaths(df=self.scanpath_df_all, data_type="all")
        else:
            print("Wrong Data Type")


if __name__ == '__main__':
    """
    Corpus
        - GoogleNQ
    AOI_Type
        - Paragraph
    which_data_type
        - All
        - Agree
        - Both
    num_node_features
        - 2
        - 17
    which_model
        - GNN1
        - GNN2
        - GNN3
    """
    number_of_optuna_trials = 150
    number_of_optuna_jobs = -1
    logging.basicConfig()
    GPU = 0
    AOI_Type = "Paragraph"
    for Data_Type in ["All", "Agree"]:
        for Model in ["GNN1", "GNN2", "GNN3"]:
            for Number_Of_Node_Features in [2, 17]:
                processes_1 = Processing(corpus="GoogleNQ",
                                         aoi_type=AOI_Type,
                                         which_data_type=Data_Type,
                                         num_node_features=Number_Of_Node_Features,
                                         which_model=Model,
                                         number_of_trials=number_of_optuna_trials,
                                         gpu=GPU,
                                         number_of_jobs=number_of_optuna_jobs)
                print(f"GoogleNQ {Data_Type}, {AOI_Type}, {Model}, {Number_Of_Node_Features} Done")
