import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from statsmodels.tsa.stattools import adfuller, acf, kpss
from matplotlib import pyplot
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer
#import xgboost as xgb
import concurrent.futures
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import skew, kurtosis
from scipy.fft import fft
import time
from datetime import timedelta
# from multiprocessing import Process, Manager, cpu_count
import multiprocessing as mp
from functools import partial
from prophet import Prophet
from colorama import init as colorama_init
from colorama import Fore
from colorama import Style
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from scipy.stats import entropy
from scipy.signal import find_peaks
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import pywt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
import arch
import random
import hurst
import ruptures as rpt
import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"
#device = "cpu"



# class LSTMClassifier(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes, num_layers=3, dropout=0.5):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0 )
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x, lengths):
#         lengths_cpu = lengths.cpu()
#         packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
#         packed_out, (ht, ct) = self.lstm(packed)
#         out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
#         out = out[torch.arange(x.size(0), device=x.device), lengths-1]
#         return self.fc(out)





X = pd.read_csv("input.csv")

y = X["labels"]

print(X["labels"].value_counts())

X = X.drop(columns=["labels"])

with open("nn_input", 'rb') as fp:
    nn_windows = pickle.load(fp)

# print(f"Original class distribution: {y.value_counts().to_dict()}")
# print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)

assert y_encoded.min() >= 0 and y_encoded.max() < 3, "Invalid class labels detected"

min_length = min(len(win) for win in nn_windows)
assert min_length > 0, "Found sequence with length <= 0"

features_dim = nn_windows[0].shape[-1]

# Split into train and test sets
indices = np.arange(len(X))
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, y_encoded, indices, test_size=0.2, random_state=42, stratify=y_encoded)


class WindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = [torch.tensor(win, dtype=torch.float32) for win in windows]
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.lengths = [len(win) for win in windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i], self.labels[i], self.lengths[i]

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_len, hidden_size*2)
        energy = torch.tanh(self.W(hidden_states))
        attention = self.V(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, attention_weights


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3, dropout=0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0 )
        self.attention = TemporalAttention(hidden_size*2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, lengths):
        lengths_cpu = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths_cpu,
                   batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        context, _ = self.attention(out)
        return self.fc(context)



def collate_fn(batch):
    # Unpack the three elements from each batch item
    sequences, labels, lengths = zip(*batch)
    # Convert lengths to a tensor
    lengths = torch.tensor(lengths)
    # Sort the batch by lengths in descending order
    sorted_indices = torch.argsort(lengths, descending=True)
    sorted_sequences = [sequences[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_lengths = lengths[sorted_indices]

    # Pad sequences to the maximum length in the batch
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sorted_sequences, batch_first=True)
    sorted_labels = torch.stack(sorted_labels)

    return padded_sequences, sorted_labels, sorted_lengths


if __name__ == "__main__":
    nn_dataset = WindowDataset(nn_windows, y_encoded)
    dataloader = DataLoader(nn_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Train Neural Network
    model = LSTMClassifier(input_size=7, hidden_size=128, num_classes=3).to(device)


    nn_X_train = [nn_windows[i] for i in indices_train]
    nn_X_test = [nn_windows[i] for i in indices_test]
    nn_y_train = y_encoded[indices_train]
    nn_y_test = y_encoded[indices_test]

    # Handle class imbalance for NN
    # class_counts = np.bincount(nn_y_train)
    # class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)

    # Create DataLoaders with correct splits
    train_dataset = WindowDataset(nn_X_train, nn_y_train)
    test_dataset = WindowDataset(nn_X_test, nn_y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    max_grad_norm = 1.0
    for epoch in range(500):
        model.train()
        running_loss = 0.0
        for inputs, labels, lengths in train_loader:
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = loss_fn(outputs, labels)

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {epoch+1}, Loss: {running_loss:.4f}")
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        if epoch % 10 == 0:
            correct = 0
            total = 0
            with torch.inference_mode():
                for inputs, labels, lengths in test_loader:
                    inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
                    outputs = model(inputs, lengths)
                    test_loss = loss_fn(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    val_loss += loss_fn(outputs, labels).item()
                print(f"Test Accuracy: {100 * correct / total:.2f}%, Validation Loss: {val_loss}")
        # Early stopping
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= 20:
        #         print("Early stopping!")
        #         break

    def get_nn_predictions(model, loader):
        model.eval()
        all_probas = []
        with torch.inference_mode():
            for inputs, _ , lengths in loader:
                inputs, lengths = inputs.to(device), lengths.to(device)
                outputs = model(inputs, lengths)
                probas = torch.softmax(outputs, dim=1).cpu()
                all_probas.append(probas.numpy())
        return np.concatenate(all_probas, axis=0)



    print(f"{Fore.CYAN}BayesSearch Parameter tuning in progress{Style.RESET_ALL}")

    # Initialize classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    gb_classifier = GradientBoostingClassifier(random_state=42)

    rf_param_space = {
        'n_estimators': Integer(100, 1000),
        'max_depth': Integer(10, 50),
        'min_samples_split': Integer(2, 20),
        'min_samples_leaf': Integer(1, 10),
        'max_features': Categorical(['sqrt', 'log2']),
        'class_weight': Categorical(['balanced', None])
    }

    # Define the parameter space for Gradient Boosting
    gb_param_space = {
        'n_estimators': Integer(100, 500),
        'learning_rate': Real(0.01, 0.2),
        'max_depth': Integer(3, 10),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 4)
    }

    # Best parameters for Random Forest:
    # OrderedDict({
    # 'class_weight': balaned,
    # 'max_depth': 12,
    # 'max_features': 'log2',
    # 'min_samples_leaf': 1,
    # 'min_samples_split': 3,
    # 'n_estimators': 152})


    # Best parameters for Gradient Boosting: OrderedDict({
    # 'learning_rate': 0.02699050939170252,
    # 'max_depth': 6,
    # 'min_samples_leaf': 4,
    # 'min_samples_split': 2,
    # 'n_estimators': 150})
    rf = RandomForestClassifier(
            n_estimators=500,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )


    class_weights = class_weight.compute_sample_weight('balanced', y_train)

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    rf_bayes_search = BayesSearchCV(
        estimator=rf_classifier,
        search_spaces=rf_param_space,
        n_iter=50,  # Number of iterations for Bayesian optimization
        cv=TimeSeriesSplit(n_splits=3),  # Use time-series cross-validation
        scoring='accuracy',  # Metric to optimize
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )

    # Set up BayesSearchCV for Gradient Boosting
    gb_bayes_search = BayesSearchCV(
        estimator=gb_classifier,
        search_spaces=gb_param_space,
        n_iter=50,  # Number of iterations for Bayesian optimization
        cv=TimeSeriesSplit(n_splits=3),  # Use time-series cross-validation
        scoring='accuracy',  # Metric to optimize
        n_jobs=-1,  # Use all available CPU cores
        random_state=42
    )

    # Fit BayesSearchCV for Random Forest
    print("Optimizing Random Forest...")
    rf_bayes_search.fit(X_train, y_train)
    print("Best parameters for Random Forest:", rf_bayes_search.best_params_)
    print("Best cross-validation score for Random Forest:", rf_bayes_search.best_score_)

    # Fit BayesSearchCV for Gradient Boosting
    print("Optimizing Gradient Boosting...")
    gb_bayes_search.fit(X_train, y_train)
    print("Best parameters for Gradient Boosting:", gb_bayes_search.best_params_)
    print("Best cross-validation score for Gradient Boosting:", gb_bayes_search.best_score_)

    # Get the best models
    best_rf = rf_bayes_search.best_estimator_
    best_gb = gb_bayes_search.best_estimator_


    # Create the VotingClassifier with the best models
    classifier = VotingClassifier(
        estimators=[
            ('rf', best_rf),
            ('gb', best_gb),
        ],
        voting='soft',  # Use soft voting for probability-based predictions
        n_jobs=-1
    )

    # Train the ensemble
    print(f"{Fore.CYAN}Training Voting Classifier{Style.RESET_ALL}")
    classifier.fit(X_train, y_train)


    nn_proba = get_nn_predictions(model, test_loader)
    vc_proba = classifier.predict_proba(X_test)

    # Combine predictions using weighted average
    combined_proba = 0.6*vc_proba + 0.4*nn_proba  # Adjust weights based on model performance
    y_pred_combined = np.argmax(combined_proba, axis=1)

    print("\nCombined Model Performance:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_combined):.4f}")
    print(classification_report(y_test, y_pred_combined))
    print(confusion_matrix(y_test, y_pred_combined))

    # Evaluate model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred_combined)
    class_report = classification_report(y_test, y_pred_combined, zero_division=1)
    cross_val_scores = cross_val_score(classifier, X, y, cv=5)

    tscv = TimeSeriesSplit(n_splits=5)
    cross_val_scores2 = cross_val_score(classifier, X, y, cv=tscv)

    print("Accuracy on test set:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)
    print("Cross Validation Score: ",cross_val_scores)
    print("Cross Validation Score 2: ",cross_val_scores2)

    # TODO
    # Save the neural network and the ensemble model
    for param in model.state_dict():
        print(param, "\t", model.state_dict()[param].size())

    with open('/home/bhargav/Documents/projects/AutoML/saved_models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(classifier, f)

    torch.save(model.state_dict(), "/home/bhargav/Documents/projects/AutoML/saved_models/nn_model.pth")
