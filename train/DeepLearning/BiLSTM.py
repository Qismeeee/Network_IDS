import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, label_binarize
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils.class_weight import compute_class_weight

# Focal Loss Definition


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# Data Preprocessing Function


def load_and_preprocess_data(root, scaler_choice='standard', apply_log_transform=True):
    NB15_1 = pd.read_csv(root + 'UNSW-NB15_1.csv', low_memory=False)
    NB15_2 = pd.read_csv(root + 'UNSW-NB15_2.csv', low_memory=False)
    NB15_3 = pd.read_csv(root + 'UNSW-NB15_3.csv', low_memory=False)
    NB15_4 = pd.read_csv(root + 'UNSW-NB15_4.csv', low_memory=False)
    NB15_features = pd.read_csv(
        root + 'NUSW-NB15_features.csv', encoding='cp1252')

    NB15_1.columns = NB15_features['Name']
    NB15_2.columns = NB15_features['Name']
    NB15_3.columns = NB15_features['Name']
    NB15_4.columns = NB15_features['Name']

    train_df = pd.concat([NB15_1, NB15_2, NB15_3, NB15_4], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df['attack_cat'] = train_df['attack_cat'].fillna(
        'normal').apply(lambda x: x.strip().lower())
    train_df['ct_flw_http_mthd'] = train_df['ct_flw_http_mthd'].fillna(0)
    train_df['is_ftp_login'] = train_df['is_ftp_login'].fillna(0)
    train_df['attack_cat'] = train_df['attack_cat'].replace(
        'backdoors', 'backdoor')

    label_mapping = {
        'normal': 6, 'analysis': 0, 'backdoor': 1, 'dos': 2, 'exploits': 3,
        'fuzzers': 4, 'generic': 5, 'reconnaissance': 7, 'shellcode': 8, 'worms': 9
    }
    train_df['attack_cat'] = train_df['attack_cat'].map(label_mapping)
    train_df = train_df.dropna(subset=['attack_cat'])
    train_df['attack_cat'] = train_df['attack_cat'].astype(int)

    numeric_cols = [
        'sport', 'dsport', 'ct_ftp_cmd', 'Ltime', 'Stime', 'sbytes', 'dbytes', 'Spkts',
        'Dpkts', 'Sload', 'Dload', 'Sjit', 'Djit', 'tcprtt', 'synack', 'ackdat'
    ]
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

    imputer = SimpleImputer(strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])

    if apply_log_transform:
        for col in numeric_cols:
            train_df[col] = np.log1p(train_df[col])

    train_df['duration'] = train_df['Ltime'] - train_df['Stime']
    train_df['byte_ratio'] = train_df['sbytes'] / (train_df['dbytes'] + 1)
    train_df['pkt_ratio'] = train_df['Spkts'] / (train_df['Dpkts'] + 1)
    train_df['load_ratio'] = train_df['Sload'] / (train_df['Dload'] + 1)
    train_df['jit_ratio'] = train_df['Sjit'] / (train_df['Djit'] + 1)
    train_df['tcp_setup_ratio'] = train_df['tcprtt'] / \
        (train_df['synack'] + train_df['ackdat'] + 1)

    columns_to_drop = ['sport', 'dsport', 'proto',
                       'srcip', 'dstip', 'state', 'service', 'swim', 'dwim', 'stcpb', 'dtcpb', 'Stime', 'Ltime']
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')

    X = train_df.drop(['attack_cat'], axis=1)
    y = train_df['attack_cat']

    iso = IsolationForest(contamination=0.01, random_state=42)
    y_pred_outliers = iso.fit_predict(X)
    mask = y_pred_outliers != -1
    X, y = X[mask], y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scaler_dict.get(scaler_choice, StandardScaler())
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2, dropout=0.1):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        # Multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x):
        # (batch_size, seq_len=1, hidden_dim)
        x = self.embedding(x).unsqueeze(1)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_dim * 2)
        x = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        x = self.layer_norm(x)  # Apply LayerNorm here
        x = self.dropout(x)
        logits = self.fc(x)  # Final output
        return logits


# Training Function
def train_epoch(model, train_loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(train_loader)
    elapsed_time = time.time() - start_time
    return avg_loss, accuracy, elapsed_time, all_preds, all_labels

# Evaluation Function


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    start_time = time.time()  # Start time for evaluation

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    elapsed_time = time.time() - start_time  # Calculate evaluation time
    return avg_loss, accuracy, elapsed_time, all_preds, all_labels


# Plot Accuracy & Loss
def plot_accuracy_loss(train_accuracies, val_accuracies, train_losses, val_losses):
    epochs = np.arange(1, len(train_accuracies) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(epochs, val_accuracies,
             label='Validation Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('BiLSTM_accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('BiLSTM_loss.png')
    plt.show()

# Plot Training & Evaluation Time


def plot_training_evaluation_time(train_times, eval_times):
    epochs = np.arange(1, len(train_times) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Time (minutes)', color=color)
    ax1.plot(epochs, train_times, label='Training Time', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Evaluation Time (seconds)', color=color)
    ax2.plot(epochs, eval_times, label='Evaluation Time', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training and Evaluation Time per Epoch")
    fig.tight_layout()  # to avoid overlap
    plt.grid(True)
    plt.savefig('BiLSTM_train_evaluation_time.png')
    plt.show()

# Plot Confusion Matrix


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.savefig('BiLSTM_confusion_matrix.png')
    plt.show()

# Plot ROC-AUC Curves


def plot_roc_auc(y_true, y_pred, num_classes):
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(num_classes))

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_pred_bin.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, color='blue',
             label=f'Micro-AUC = {roc_auc_micro:.4f}')

    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} AUC = {roc_auc[i]:.4f}')

    plt.plot([0, 1], [0, 1], linestyle='--', color='grey')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves')
    plt.legend(loc='best')
    plt.grid(True)
    plt.savefig('BiLSTM_roc_auc.png')
    plt.show()


def main():
    root = 'data/'
    batch_size = 64
    num_epochs = 1
    learning_rate = 0.001
    scaler_choice = 'standard'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(
        root, scaler_choice)

    input_dim = X_train_scaled.shape[1]
    num_classes = 10
    classes = ['analysis', 'backdoor', 'dos', 'exploits', 'fuzzers',
               'generic', 'normal', 'reconnaissance', 'shellcode', 'worms']

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                  torch.tensor(y_train.values, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32),
                                 torch.tensor(y_test.values, dtype=torch.long))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    model = BiLSTMClassifier(
        input_dim=input_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = FocalLoss()

    # Initialize lists to store metrics
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    train_times = []
    eval_times = []

    total_start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy, train_time, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, criterion, None, device
        )
        val_loss, val_accuracy, eval_time, val_preds, val_labels = evaluate_model(
            model, test_loader, criterion, device
        )

        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_times.append(train_time)
        eval_times.append(eval_time)

        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_accuracy:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | Train Time: {
                  train_time:.2f} seconds | "
              f"Eval Time: {eval_time:.2f} seconds")

    total_time = (time.time() - total_start_time) / 60
    print(f"Total Training Time: {total_time:.2f} minutes")

    plot_accuracy_loss(train_accuracies, val_accuracies,
                       train_losses, val_losses)
    plot_training_evaluation_time(train_times, eval_times)
    plot_confusion_matrix(val_labels, val_preds, classes)
    plot_roc_auc(val_labels, val_preds, num_classes)

    print("Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=classes))
    print("Confusion Matrix:")
    ConfusionMatrixDisplay(confusion_matrix(val_labels, val_preds)).plot()


if __name__ == '__main__':
    main()
