import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from sklearn.metrics import f1_score, balanced_accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class VGG19BinaryClassifier(nn.Module):
    def __init__(self, num_classes=1, dropout_prob=0.2):
        super(VGG19BinaryClassifier, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        for param in vgg19.parameters():
            param.requires_grad = False

        vgg19.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.model = vgg19

    def forward(self, x):
        return self.model(x)


class ScanpathDataset(Dataset):
    def __init__(self, events_df, transform=None):
        self.transform = transform
        self.events_df = events_df

    def __len__(self):
        return len(self.events_df)

    def __getitem__(self, idx):
        img_path = self.events_df.loc[idx]["img_path"]
        img = Image.open(img_path).convert("RGB")
        pr_label = self.events_df.loc[idx]["label"]

        if self.transform:
            img = self.transform(img)

        return img, pr_label


class Processing:
    def __init__(self, corpus: str, data_type: str, number_of_trials: int, number_of_jobs: int, gpu: int, transform):
        super().__init__()
        self.corpus = corpus
        self.data_type = data_type
        self.transform = transform
        self.dataset_dict = self.load_data()
        self.device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        print(f"Device is: {self.device}")

        self.full_training_subset = None
        self.number_of_trials = number_of_trials
        self.number_of_jobs = number_of_jobs
        self.epoch = [30, 40, 50, 100, 200]
        self.learning_rate = [0.1, 0.01, 0.001, 0.0001]
        self.batch_size = [8, 16, 32, 64]
        self.dropout_prob = [0.1, 0.2, 0.3]

    def load_data(self):
        """
        Corpus
            - gREL
            - GoogleNQ
        """
        df_all = pd.read_csv(f"../Data/CNN_Data/{self.corpus}_event_data.csv")

        # Define transformations
        if self.corpus == "gREL":
            df_agree = df_all.loc[df_all["label"] == df_all["system_label"]]
            df_agree = df_agree.loc[df_agree["gREL_label"] != "t"].reset_index(drop=True)
            df_topical = df_all.loc[df_all["gREL_label"] == "t"].reset_index(drop=True)
            dict_data = {"all": ScanpathDataset(events_df=df_all, transform=self.transform),
                         "agree": ScanpathDataset(events_df=df_agree, transform=self.transform),
                         "topical": ScanpathDataset(events_df=df_topical, transform=self.transform)}
        else:
            df_agree = df_all.loc[df_all["label"] == df_all["system_label"]].reset_index(drop=True)
            dict_data = {"all": ScanpathDataset(events_df=df_all, transform=self.transform),
                         "agree": ScanpathDataset(events_df=df_agree, transform=self.transform)}

        return dict_data

    def evaluate_results(self, true_labels, prediction_labels):
        accuracy = balanced_accuracy_score(true_labels, prediction_labels)
        loss_score = log_loss(true_labels, prediction_labels)
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

        return f1_result, accuracy, loss_score, tpr, fpr, tn, fp, fn, tp

    def train_evaluate_model(self, epoch, dropout_prob, learning_rate, training_loader, evaluation_loader):
        # Create the VGG19 binary classifier model
        model = VGG19BinaryClassifier(num_classes=1, dropout_prob=dropout_prob).to(self.device)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # Train the neural network
        model.train().to(self.device)
        train_true = []
        train_pred = []
        for epoch_run in range(epoch):
            for images, labels in training_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                loss.backward()
                optimizer.step()
                if epoch_run == epoch - 1:
                    predictions = (outputs > 0.5).float()
                    train_true.extend(labels.cpu().numpy())
                    train_pred.extend(predictions.cpu().numpy())
            # print(f"Epoch {epoch_run} loss: {loss.item()}")

        # Evaluate the trained neural network
        model.eval().to(self.device)
        with torch.no_grad():
            evaluation_true = []
            evaluation_pred = []
            for images, labels in evaluation_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                predictions = (outputs > 0.5).float()
                evaluation_true.extend(labels.cpu().numpy())
                evaluation_pred.extend(predictions.cpu().numpy())

        return train_true, train_pred, evaluation_true, evaluation_pred

    def optuna_optimise_hyperparameters(self, trial):
        # Define the hyperparameters
        epoch = trial.suggest_categorical("epoch", self.epoch)
        learning_rate = trial.suggest_categorical("learning_rate", self.learning_rate)
        batch_size = trial.suggest_categorical("batch_size", self.batch_size)
        dropout_prob = trial.suggest_categorical("dropout_prob", self.dropout_prob)

        list_validation_accuracy = []

        # Pick the index of the data splits based on the groups
        inner_loop_fold_splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=256)
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

        for fold_num, (training_index, val_index) in enumerate(
                inner_loop_fold_splits.split(self.full_training_subset, self.full_training_subset["label"],
                                             self.full_training_subset["user_id"])):

            dataframe_training = self.full_training_subset.iloc[training_index].reset_index(drop=True)
            dataframe_val = self.full_training_subset.iloc[val_index].reset_index(drop=True)
            # print(dataframe_training)

            training_dataset = ScanpathDataset(dataframe_training, transform=self.transform)
            val_dataset = ScanpathDataset(dataframe_val, transform=self.transform)

            training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            train_true, train_pred, val_true, val_pred = self.train_evaluate_model(epoch=epoch,
                                                                                   learning_rate=learning_rate,
                                                                                   dropout_prob=dropout_prob,
                                                                                   training_loader=training_loader,
                                                                                   evaluation_loader=val_loader)

            val_f1_result, val_accuracy, val_loss_score, val_tpr, val_fpr, val_tn, val_fp, val_fn, val_tp = self.evaluate_results(
                true_labels=val_true, prediction_labels=val_pred)

            list_validation_accuracy.append(val_accuracy)

            trial.report(val_accuracy, fold_num)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        average_validation_accuracy = np.sum(list_validation_accuracy) / len(list_validation_accuracy)

        return average_validation_accuracy

    def process_scanpaths(self):
        results = []
        dataset = self.dataset_dict[self.data_type]

        outer_loop_fold_iterator = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=256)
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

        # Outer cross-validation loop for the testing data
        for fold_number, (training_val_index, testing_index) in enumerate(
                outer_loop_fold_iterator.split(dataset.events_df,
                                               dataset.events_df["label"],
                                               dataset.events_df["user_id"])):
            self.full_training_subset = dataset.events_df.iloc[training_val_index]
            testing_subset = dataset.events_df.iloc[testing_index]

            unique_testing_user_ids = testing_subset["user_id"].unique().tolist()
            # Optimise hyperparameters
            study = optuna.create_study(direction="maximize")
            number_of_trials = self.number_of_trials
            study.optimize(self.optuna_optimise_hyperparameters, n_trials=number_of_trials, n_jobs=self.number_of_jobs)
            print("Best trial:")
            best_trial = study.best_trial
            print("\t Value: ", best_trial.value)
            print(best_trial.params)

            training_dataset = ScanpathDataset(self.full_training_subset, transform=self.transform)
            testing_dataset = ScanpathDataset(testing_subset, transform=self.transform)

            training_loader = DataLoader(training_dataset, batch_size=best_trial.params["batch_size"], shuffle=True)
            testing_loader = DataLoader(testing_dataset, batch_size=best_trial.params["batch_size"], shuffle=False)

            # Train and Test Model
            train_true, train_pred, test_true, test_pred = self.train_evaluate_model(epoch=best_trial.params["epoch"],
                                                                                     learning_rate=best_trial.params[
                                                                                         "learning_rate"],
                                                                                     dropout_prob=best_trial.params[
                                                                                         "dropout_prob"],
                                                                                     training_loader=training_loader,
                                                                                     evaluation_loader=testing_loader)

            # Calculate evaluation metrics for the current fold
            training_f1_result, training_accuracy, training_loss_score, training_tpr, training_fpr, training_tn, training_fp, training_fn, training_tp = self.evaluate_results(
                true_labels=test_true, prediction_labels=test_pred)
            testing_f1_result, testing_accuracy, testing_loss_score, testing_tpr, testing_fpr, testing_tn, testing_fp, testing_fn, testing_tp = self.evaluate_results(
                true_labels=test_true, prediction_labels=test_pred)

            results.append({"user_id": unique_testing_user_ids,
                            "epoch": best_trial.params["epoch"],
                            "learning_rate": best_trial.params["learning_rate"],
                            "batch_size": best_trial.params["batch_size"],
                            "dropout_prob": best_trial.params["dropout_prob"],
                            "f1_score_testing": testing_f1_result,
                            "accuracy_testing": testing_accuracy, "tpr_testing": testing_tpr,
                            "fpr_testing": testing_fpr,
                            "loss_testing": testing_loss_score, "tn_testing": testing_tn, "fp_testing": testing_fp,
                            "fn_testing": testing_fn,
                            "tp_testing": testing_tp,
                            "f1_score_training": training_f1_result,
                            "accuracy_training": training_accuracy, "tpr_training": training_tpr,
                            "fpr_training": training_fpr,
                            "loss_training": training_loss_score, "tn_training": training_tn,
                            "fp_training": training_fp,
                            "fn_training": training_fn,
                            "tp_training": training_tp
                            })
        return results

    def run(self):
        results_list = self.process_scanpaths()
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(f'../Results/CNN/{self.corpus}_{self.data_type}_VGG19.csv', index=False)


if __name__ == '__main__':
    """
    Corpus
        - gREL
        - GoogleNQ
    data_type
        - all
        - agree
        - topical
    """
    number_of_optuna_trials = 100
    number_of_optuna_jobs = -1
    GPU = 0
    torch.multiprocessing.set_start_method("spawn")
    transformation = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), ])

    Corpus = "GoogleNQ"
    for DataType in ["all", "agree"]:
        processes = Processing(corpus=Corpus, data_type=DataType, number_of_trials=number_of_optuna_trials,
                               transform=transformation, number_of_jobs=number_of_optuna_jobs, gpu=GPU)
        processes.run()
        print(f"{Corpus} {DataType} Done")

    Corpus = "gREL"
    for DataType in ["all", "agree", "topical"]:
        processes = Processing(corpus=Corpus, data_type=DataType, number_of_trials=number_of_optuna_trials,
                               transform=transformation, number_of_jobs=number_of_optuna_jobs, gpu=GPU)
        processes.run()
        print(f"{Corpus} {DataType} Done")
