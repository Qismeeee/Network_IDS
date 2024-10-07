import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../preprocess'))
from preprocess import load_and_preprocess_data  

def save_confusion_matrix(y_test, y_pred, model_name, save_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()

def save_metrics_plot(y_test, y_pred, model_name, save_dir):
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']

    for metric in metrics:
        values = [v[metric] for k, v in report_dict.items() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        classes = list(report_dict.keys())[:-3]  

        plt.figure(figsize=(10, 6))
        plt.bar(classes, values)
        plt.title(f"{metric.capitalize()} per Class - {model_name}")
        plt.xlabel('Classes')
        plt.ylabel(metric.capitalize())
        plt.xticks(rotation=45)
        plt.tight_layout()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{model_name}_{metric}.png'))
        plt.close()

def run_random_forest(X_train, X_test, y_train, y_test):
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=None, 
        random_state=42, 
        n_jobs=-1,  
        class_weight='balanced_subsample'  
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    start_time = time.time()
    y_pred = model.predict(X_test)
    evaluation_time = time.time() - start_time
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Training time: {training_time:.4f} seconds")
    print(f"Evaluation time: {evaluation_time:.4f} seconds")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    report = classification_report(y_test, y_pred)
    print(report)

    save_dir = "train/MachineLearning/Picture/"
    save_confusion_matrix(y_test, y_pred, "RandomForest", save_dir)
    save_metrics_plot(y_test, y_pred, "RandomForest", save_dir)

def main():
    root = 'preprocess/'  
    X_train_scaled, X_test_scaled, y_train_resampled, y_test = load_and_preprocess_data(root='data/')
    run_random_forest(X_train_scaled, X_test_scaled, y_train_resampled, y_test)

if __name__ == "__main__":
    main()
