import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import math
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# binary classification
def binary_plot_bar_distribution(y,chartname):
    plt.figure(figsize=(8, 5))
    sns.countplot(x=y)
    plt.title(chartname)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Attack'])
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

def binary_plot_pie_distribution(y,chartname):
    unique, counts = np.unique(y, return_counts=True)
    labels = ['Normal' if label == 0 else 'Attack' for label in unique]
    plt.figure(figsize=(8, 5))
    plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
    plt.axis('equal')
    plt.title(chartname)
    plt.show()

def binary_plot_table_distribution(y,chartname):
    unique, counts = np.unique(y, return_counts=True)
    labels = ['Normal' if label == 0 else 'Attack' for label in unique]
    table_data = pd.DataFrame({'type': labels, 'Number': counts})
    plt.figure(figsize=(6, 2))
    plt.tight_layout()
    plt.axis('off')
    plt.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title(chartname)
    plt.show()

#multi classification
def plot_attack_category_bar(y,chartname):
    attack_labels = {
        0: 'analysis', 1: 'backdoor', 2: 'dos', 3: 'exploits',
        4: 'fuzzers', 5: 'generic', 6: 'normal', 7: 'reconnaissance', 
        8: 'shellcode', 9: 'worms'
    }
    y_mapped = [attack_labels[label] for label in y]
    df = pd.DataFrame({'Attack Type': y_mapped})
    attack_counts = df['Attack Type'].value_counts().reset_index()
    attack_counts.columns = ['Attack Type', 'Sample']

    # Tạo biểu đồ
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y_mapped, order=attack_counts['Attack Type'], palette="viridis")
    plt.title(chartname)
    plt.xlabel('Categories')
    plt.xticks(rotation=40)
    plt.ylabel('Sample')
    plt.ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    plt.show()

def plot_attack_category_table(y,chartname):
    attack_labels = {
        0: 'analysis', 1: 'backdoor', 2: 'dos', 3: 'exploits',
        4: 'fuzzers', 5: 'generic', 6: 'normal', 7: 'reconnaissance', 
        8: 'shellcode', 9: 'worms'
    }
    y_mapped = [attack_labels[label] for label in y]
    df = pd.DataFrame({'Attack Type': y_mapped})
    attack_counts = df['Attack Type'].value_counts().reset_index()
    attack_counts.columns = ['Attack Type', 'Sample']

    # Hiển thị bảng
    plt.figure(figsize=(6, 4))
    plt.title(chartname)
    column_labels = ['Attack Type', 'Sample']
    cell_text = attack_counts.values.tolist()

    table = plt.table(cellText=cell_text, colLabels=column_labels, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Ẩn các trục của bảng
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_attack_category_table_train_test(y_train, y_test, chartname):

    attack_labels = {
        0: 'analysis', 1: 'backdoor', 2: 'dos', 3: 'exploits',
        4: 'fuzzers', 5: 'generic', 6: 'normal', 7: 'reconnaissance', 
        8: 'shellcode', 9: 'worms'
    }
    y_train_mapped = [attack_labels[label] for label in y_train]
    y_test_mapped = [attack_labels[label] for label in y_test]
    
    train_counts = pd.Series(y_train_mapped).value_counts().reindex(attack_labels.values(), fill_value=0)
    test_counts = pd.Series(y_test_mapped).value_counts().reindex(attack_labels.values(), fill_value=0)
    sample_df_train = pd.DataFrame({'Attack Type': train_counts.index, 'Train Sample': train_counts.values})
    sample_df_test = pd.DataFrame({'Attack Type': test_counts.index, 'Test Sample': test_counts.values})

    combined_df = pd.merge(sample_df_train, sample_df_test, on='Attack Type', how='outer')

    total_train = train_counts.sum()
    total_test = test_counts.sum()

    total_row = pd.DataFrame({'Attack Type': ['Total'], 'Train Sample': [total_train], 'Test Sample': [total_test]})

    combined_df = pd.concat([combined_df, total_row], ignore_index=True)

    combined_df[:-1].sort_values(by='Train Sample', ascending=False, inplace=True)  # Sắp xếp ngoại trừ hàng cuối

    plt.figure(figsize=(8, 5))
    plt.title(chartname)
    column_labels = ['Attack Type', 'Train Sample', 'Test Sample']

    table = plt.table(cellText=combined_df.values, colLabels=column_labels, cellLoc='center', loc='center')

    for i in range(len(combined_df)):
        color = "lightgrey" if i % 2 == 0 else "white"
        table[(i + 1, 0)].set_facecolor(color)  # Màu cho cột "Attack Type"
        table[(i + 1, 1)].set_facecolor(color)  # Màu cho cột "Train Sample"
        table[(i + 1, 2)].set_facecolor(color)  # Màu cho cột "Test Sample"

    plt.axis('off')

    plt.tight_layout()
    plt.show()



# Hàm vẽ biểu đồ hộp của các cột số
def plot_boxplots(train_df):
    numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_columns = len(numerical_columns)

    # Tính số lượng hàng và cột cho subplot
    num_cols = 3
    num_rows = math.ceil(num_columns / num_cols)
    plt.figure(figsize=(18, num_rows * 3))
    
    sns.set_palette("husl")
    sns.set(style="whitegrid")
    
    for i, col in enumerate(numerical_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.boxplot(x=train_df[col], color='skyblue', width=0.5)
        plt.title(col)
        plt.xlabel("")
    
    plt.suptitle("Distribution of Key Medical Indicators", y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


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

    X = train_df.drop(['attack_cat'], axis=1)
    y = train_df['attack_cat']

    plot_boxplots(train_df)
    iso = IsolationForest(contamination=0.01, random_state=42)
    y_pred_outliers = iso.fit_predict(X)
    mask = y_pred_outliers != -1
    X, y = X[mask], y[mask]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    plot_attack_category_table_train_test(y_train, y_test, "Số Lượng Mẫu Từng Class Trong Tập Train và Test")

    plot_attack_category_bar(y_train,"Số Lượng Mẫu Từng Class Trước Khi Tái Lấy Mẫu")
    plot_attack_category_table(y_train,"Số Lượng Mẫu Từng Class Trước Khi Tái Lấy Mẫu")

    X_train_resampled, y_train_resampled = handle_resampling(X_train, y_train)

    
    plot_attack_category_bar(y_train_resampled,"Số Lượng Mẫu Từng Class Sau Khi Tái Lấy Mẫu")
    plot_attack_category_table(y_train_resampled,"Số Lượng Mẫu Từng Class Sau Khi Tái Lấy Mẫu")

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


load_and_preprocess_data("D:/Download/UNSW-NB15/", scaler_choice='standard')