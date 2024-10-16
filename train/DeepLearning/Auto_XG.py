import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
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


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        # Giải mã về đúng kích thước ban đầu (input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            # Kích thước cuối cùng phải là input_dim
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        # Kết quả giải mã cần có kích thước bằng input_dim
        decoded = self.decoder(encoded)
        return encoded, decoded


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
    plt.savefig('autoencoder_xgboost_accuracy.png')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('autoencoder_xgboost_loss.png')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.grid(False)
    plt.savefig('autoencoder_xgboost_confusion_matrix.png')
    plt.show()


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
    plt.savefig('autoencoder_xgboost_roc_auc.png')
    plt.show()


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
    plt.savefig('transformer_train_evaluation_time.png')
    plt.show()


def main():
    root = "data/"
    hidden_dim = 128
    batch_size = 512
    num_epochs = 100
    learning_rate = 1e-3

    # Load và tiền xử lý dữ liệu
    X_train, X_test, y_train, y_test = load_and_preprocess_data(root)

    input_dim = X_train.shape[1]

    # Bước 1: Train Autoencoder để trích xuất đặc trưng
    autoencoder = Autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
    autoencoder_optimizer = torch.optim.Adam(
        autoencoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    train_accuracies, val_accuracies = [], []
    train_losses, val_losses = [], []
    train_times, eval_times = []  # To store times for each epoch

    print("Training Autoencoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    for epoch in range(3):
        start_time = time.time()  # Start tracking time for training
        autoencoder_optimizer.zero_grad()
        encoded, decoded = autoencoder(X_train_tensor)  # Cần lấy cả 'decoded'
        # Tính toán MSELoss giữa đầu ra được giải mã và đầu vào
        train_loss = criterion(decoded, X_train_tensor)
        train_loss.backward()
        autoencoder_optimizer.step()

        # Time for training in the current epoch
        train_time = time.time() - start_time
        train_times.append(train_time)  # Store the training time

        # Trích xuất đặc trưng từ Autoencoder sau epoch
        autoencoder.eval()
        with torch.no_grad():
            X_train_encoded = autoencoder.encoder(X_train_tensor).numpy()
            X_test_encoded = autoencoder.encoder(X_test_tensor).numpy()

        # Bước 2: Huấn luyện XGBoost với đặc trưng đã trích xuất từ Autoencoder
        start_eval_time = time.time()  # Start tracking time for evaluation
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
        xgb_model.fit(X_train_encoded, y_train)

        # Dự đoán cho train và test
        y_train_pred = xgb_model.predict(X_train_encoded)
        y_test_pred = xgb_model.predict(X_test_encoded)

        # Time for evaluation in the current epoch
        eval_time = time.time() - start_eval_time
        eval_times.append(eval_time)  # Store the evaluation time

        # Tính toán độ chính xác và loss
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        train_loss_value = np.mean(
            (y_train_pred - y_train) ** 2)  # Tính loss thủ công
        test_loss_value = np.mean(
            (y_test_pred - y_test) ** 2)  # Tính loss thủ công

        end_time = time.time()

        # Lưu lại kết quả cho từng epoch
        train_accuracies.append(train_accuracy)
        val_accuracies.append(test_accuracy)
        train_losses.append(train_loss_value)
        val_losses.append(test_loss_value)

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {
                  test_accuracy:.4f} | "
              f"Train Loss: {train_loss_value:.4f} | Test Loss: {
                  test_loss_value:.4f} | "
              f"Training Time: {train_time:.2f} seconds | Evaluation Time: {eval_time:.2f} seconds")

    # Sau khi hoàn thành toàn bộ quá trình huấn luyện, hiển thị báo cáo phân loại và các biểu đồ
    print("\nTraining completed. Generating reports and plots...")

    # Classification Report cho tập test
    print("\nClassification Report for Test Set:")
    print(classification_report(y_test, y_test_pred))

    # Vẽ các biểu đồ
    plot_accuracy_loss(train_accuracies, val_accuracies,
                       train_losses, val_losses)
    plot_confusion_matrix(y_test, y_test_pred, classes=np.unique(y_train))
    plot_roc_auc(y_test, y_test_pred, num_classes=len(np.unique(y_train)))

    plot_training_evaluation_time(train_times, eval_times)
    print("All tasks completed.")


if __name__ == "__main__":
    main()
