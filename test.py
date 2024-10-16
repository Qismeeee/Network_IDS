import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import math
# Save plot function


def save_plot(fig, filename, save_dir='plots', dpi=300):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fig.savefig(os.path.join(save_dir, filename), dpi=dpi, bbox_inches='tight')

# Load and preprocess data


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
                       'srcip', 'dstip', 'state', 'service']
    train_df = train_df.drop(columns=columns_to_drop, errors='ignore')

    X = train_df.drop(['attack_cat'], axis=1)
    y = train_df['attack_cat']

    return train_df, X, y

# Plot skewness distribution


def plot_skewness(train_df, save_path='skewness_distribution.png'):
    numeric_columns = train_df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    num_columns = len(numeric_columns)
    num_cols = 3
    num_rows = math.ceil(num_columns / num_cols)

    plt.figure(figsize=(18, num_rows * 3))
    sns.set_palette("husl")
    sns.set(style="whitegrid")

    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(train_df[col], kde=True, color='purple')
        skewness = train_df[col].skew()
        plt.title(f'{col} (Skewness: {skewness:.2f})')
        plt.xlabel("Value")

    plt.suptitle("Skewness Distribution of Key Numerical Features",
                 y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(plt.gcf(), save_path)
    plt.show()

# Plot outliers using boxplots


def plot_outliers(train_df, save_path='outliers_boxplots.png'):
    numeric_columns = train_df.select_dtypes(
        include=['float64', 'int64']).columns.tolist()
    num_columns = len(numeric_columns)
    num_cols = 3
    num_rows = math.ceil(num_columns / num_cols)

    plt.figure(figsize=(18, num_rows * 3))
    sns.set_palette("husl")
    sns.set(style="whitegrid")

    for i, col in enumerate(numeric_columns, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.boxplot(x=train_df[col], color='skyblue')
        plt.title(f'Outliers in {col}')

    plt.suptitle("Boxplots Showing Outliers in Numerical Features",
                 y=1.02, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(plt.gcf(), save_path)
    plt.show()

# Main function


def main():
    root = 'data/'
    train_df, X, y = load_and_preprocess_data(root)

    # Plot skewness
    plot_skewness(train_df, save_path='skewness_distribution.png')

    # Plot outliers
    plot_outliers(train_df, save_path='outliers_boxplots.png')


if __name__ == "__main__":
    main()
