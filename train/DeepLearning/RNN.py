import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, label_binarize
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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

# Focal Loss Definition


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets.data.view(-1))
            F_loss = alpha_t * F_loss

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


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(
            input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(self.dropout(x[:, -1, :]))  # Output from the last timestep
        return x


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_time = time.time() - start_time
    accuracy = correct / total
    return total_loss / len(train_loader), accuracy, epoch_time


def validate_epoch(model, test_loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    eval_time = time.time() - start_time
    accuracy = correct / total
    return val_loss / len(test_loader), accuracy, eval_time, all_preds, all_labels


def plot_accuracy_loss(train_accuracies, val_accuracies, train_losses, val_losses):
    epochs = np.arange(1, len(train_accuracies) + 1)

    # Plot accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("RNN_acc.png")
    plt.show()

    # Plot loss
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("RNN_loss.png")
    plt.show()


def plot_confusion_matrix(test_labels, test_preds, classes):
    cm = confusion_matrix(test_labels, test_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.grid(True)
    plt.savefig("RNN_confusion_matrix.png")
    plt.show()


def plot_roc_auc(test_labels, test_preds, num_classes):
    test_labels_bin = label_binarize(
        test_labels, classes=np.arange(num_classes))
    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_bin[:, i], test_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('Receiver Operating Characteristic')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig("RNN_roc.png")
    plt.show()


def evaluate_model(test_labels, test_preds):
    report = classification_report(test_labels, test_preds, target_names=[
                                   f'Class {i}' for i in range(len(set(test_labels)))])
    print(report)


# Main Execution
if __name__ == '__main__':
    # Load and preprocess the dataset
    root = 'data/'  # Update with your dataset path
    X_train, X_test, y_train, y_test = load_and_preprocess_data(root)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(
        1)  # Add a dimension for sequence length
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Define model parameters
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = len(set(y_train))
    num_layers = 2
    dropout = 0.3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model, loss function, and optimizer
    model = RNNModel(input_dim, hidden_dim, output_dim,
                     num_layers, dropout).to(device)
    criterion = FocalLoss(alpha=None, gamma=2, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training and validation parameters
    num_epochs = 100  # Set your desired number of epochs
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_times, eval_times = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy, train_time = train_epoch(
            model, train_loader, optimizer, criterion, device)
        val_loss, val_accuracy, eval_time, val_preds, val_labels = validate_epoch(
            model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        train_times.append(train_time)
        eval_times.append(eval_time)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {
                  train_accuracy:.4f}, "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}",
    f"Training Time: {train_time:.2f} seconds | Evaluation Time: {eval_time:.2f} seconds")

    # Evaluate the model on the test set
    final_val_loss, final_val_accuracy, _, test_preds, test_labels = validate_epoch(
        model, test_loader, criterion, device)

    # Plotting
    plot_accuracy_loss(train_accuracies, val_accuracies,
                       train_losses, val_losses)
    plot_confusion_matrix(test_labels, test_preds, classes=range(output_dim))

    # Final evaluation
    evaluate_model(test_labels, test_preds)
