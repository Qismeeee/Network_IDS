import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages
import time


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, from_logits=False, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE, name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        cross_entropy_loss = -y_true * tf.math.log(y_pred)
        focal_loss = self.alpha * \
            tf.pow(1 - y_pred, self.gamma) * cross_entropy_loss
        return tf.reduce_sum(focal_loss, axis=1)


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

    return X_train_scaled, X_test_scaled, y_train, y_test, label_mapping


root = "data/"  
X_train_scaled, X_test_scaled, y_train_resampled, y_test, label_mapping = load_and_preprocess_data(
    root)

X_train_reshaped = np.reshape(
    X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(
    X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

num_classes = len(np.unique(y_train_resampled))
y_train_categorical = tf.keras.utils.to_categorical(
    y_train_resampled, num_classes=num_classes)
y_test_categorical = tf.keras.utils.to_categorical(
    y_test, num_classes=num_classes)

model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(
    X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss=FocalLoss(
    from_logits=False), metrics=['accuracy'])

start_time = time.time()
history = model.fit(X_train_reshaped, y_train_categorical, epochs=100,
                    batch_size=128, validation_data=(X_test_reshaped, y_test_categorical))

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

pdf_path = "training_plots.pdf"
with PdfPages(pdf_path) as pdf:
    # Plot accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    pdf.savefig()  # saves the current figure into a pdf page
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    pdf.savefig()  
    plt.close()

    y_pred = model.predict(X_test_reshaped)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes,
          target_names=[str(i) for i in range(num_classes)]))

    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[
                                  str(i) for i in range(num_classes)])
    disp.plot()
    plt.title("Confusion Matrix")
    pdf.savefig() 
    plt.close()

    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_test_categorical[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.title('AUC-ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    pdf.savefig() 
    plt.close()

print(f"All plots saved to {pdf_path}.")
