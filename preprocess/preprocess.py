import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def handle_resampling(X_train, y_train):
    desired_count = int(y_train.value_counts().median())
    
    oversample_strategy = {i: desired_count for i in range(len(y_train.value_counts())) if y_train.value_counts()[i] < desired_count}
    undersample_strategy = {i: desired_count for i in range(len(y_train.value_counts())) if y_train.value_counts()[i] > desired_count}
    
    smote = SMOTE(sampling_strategy=oversample_strategy, random_state=42)
    undersample = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
    pipeline = Pipeline(steps=[('smote', smote), ('undersample', undersample)])

    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
    
    print("Số lượng mẫu sau khi tái lấy mẫu:", y_resampled.value_counts())
    return X_resampled, y_resampled

def load_and_preprocess_data(root, scaler_choice='standard', apply_log_transform=False, apply_boxcox=False):
    NB15_1 = pd.read_csv(root + 'UNSW-NB15_1.csv', low_memory=False)
    NB15_2 = pd.read_csv(root + 'UNSW-NB15_2.csv', low_memory=False)
    NB15_3 = pd.read_csv(root + 'UNSW-NB15_3.csv', low_memory=False)
    NB15_4 = pd.read_csv(root + 'UNSW-NB15_4.csv', low_memory=False)
    NB15_features = pd.read_csv(root + 'NUSW-NB15_features.csv', encoding='cp1252')

    NB15_1.columns = NB15_features['Name']
    NB15_2.columns = NB15_features['Name']
    NB15_3.columns = NB15_features['Name']
    NB15_4.columns = NB15_features['Name']

    train_df = pd.concat([NB15_1, NB15_2, NB15_3, NB15_4], ignore_index=True)
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset
    null_columns_before = train_df.isnull().sum().sum()
    print(f"Số lượng cột có giá trị null trước khi tiền xử lý: {null_columns_before}")

    print(f"Shape trước khi tiền xử lý: {train_df.shape}")
    train_df['attack_cat'].fillna('normal', inplace=True)
    train_df['attack_cat'] = train_df['attack_cat'].apply(lambda x: x.strip().lower())
    train_df['ct_flw_http_mthd'].fillna(0, inplace=True)
    train_df['is_ftp_login'].fillna(0, inplace=True)
    train_df['attack_cat'] = train_df['attack_cat'].replace('backdoors', 'backdoor')

    label_mapping = {
        'analysis': 0, 'backdoor': 1, 'dos': 2, 'exploits': 3,
        'fuzzers': 4, 'generic': 5, 'normal': 6, 'reconnaissance': 7, 'shellcode': 8, 'worms': 9
    }
    train_df['attack_cat'] = train_df['attack_cat'].map(label_mapping)
    train_df.dropna(subset=['attack_cat'], inplace=True)

    numeric_cols = ['sport', 'dsport', 'ct_ftp_cmd', 'Ltime', 'Stime', 'sbytes', 'dbytes', 'Spkts', 
                    'Dpkts', 'Sload', 'Dload', 'Sjit', 'Djit', 'tcprtt', 'synack', 'ackdat']
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

    imputer = SimpleImputer(strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    null_columns_after = train_df.isnull().sum().sum()
    print(f"Số lượng cột có giá trị null sau khi tiền xử lý: {null_columns_after}")

    if apply_log_transform:
        for col in numeric_cols:
            train_df[col] = np.log1p(train_df[col])

    if apply_boxcox:
        power_transformer = PowerTransformer(method='box-cox', standardize=False)
        for col in numeric_cols:
            train_df[col] = np.where(train_df[col] > 0, train_df[col], 1e-6)
        train_df[numeric_cols] = power_transformer.fit_transform(train_df[numeric_cols])

    train_df['duration'] = train_df['Ltime'] - train_df['Stime']
    train_df['byte_ratio'] = train_df['sbytes'] / (train_df['dbytes'] + 1)
    train_df['pkt_ratio'] = train_df['Spkts'] / (train_df['Dpkts'] + 1)
    train_df['load_ratio'] = train_df['Sload'] / (train_df['Dload'] + 1)
    train_df['jit_ratio'] = train_df['Sjit'] / (train_df['Djit'] + 1)
    train_df['tcp_setup_ratio'] = train_df['tcprtt'] / (train_df['synack'] + train_df['ackdat'] + 1)

    columns_to_drop = ['sport', 'dsport', 'proto', 'srcip', 'dstip', 'state', 'service']
    train_df.drop(columns=columns_to_drop, inplace=True)

    print(f"Shape trước khi tái lấy mẫu: X = {train_df.drop(['attack_cat'], axis=1).shape}, y = {train_df['attack_cat'].shape}")
    print("Số lượng mẫu từng class trước khi tái lấy mẫu:")
    print(train_df['attack_cat'].value_counts())

    X = train_df.drop(['attack_cat'], axis=1)
    y = train_df['attack_cat']

    iso = IsolationForest(contamination=0.01, random_state=42)
    y_pred_outliers = iso.fit_predict(X)
    mask = y_pred_outliers != -1
    X, y = X[mask], y[mask]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_resampled, y_train_resampled = handle_resampling(X_train, y_train)

    print(f"Shape sau khi tái lấy mẫu: X_train = {X_train_resampled.shape}, y_train = {y_train_resampled.shape}")

    scaler_dict = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scaler_dict.get(scaler_choice, StandardScaler())
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test

# Hàm main
def main():
    root = 'data/'  
    X_train_scaled, X_test_scaled, y_train_resampled, y_test = load_and_preprocess_data(root, scaler_choice='standard')

if __name__ == "__main__":
    main()
