import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

# Define the DNN model
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc4 = nn.Linear(hidden_size // 4, 10)  # 10 output classes for 10 types of attacks

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return self.fc4(x)  # No softmax here

# Function to train the model
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()  # Start timing the epoch

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        # Calculate val_loss, val_accuracy, precision, recall, and F1-score
        val_loss, val_accuracy, val_precision, val_recall, val_f1_score = evaluate_model(model, test_loader, criterion, device)

        # Save the loss and accuracy values
        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1_scores.append(val_f1_score)

        # End timing the epoch
        end_time = time.time()
        epoch_time = end_time - start_time

        # Print training information
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'{len(train_loader)}/{len(train_loader)} [==============================] - '
              f'{epoch_time:.2f}s - loss: {train_losses[-1]:.4e} - '
              f'accuracy: {train_accuracies[-1]:.4f}% - '
              f'val_loss: {val_loss:.4e} - val_accuracy: {val_accuracy:.4f} - '
              f'precision: {val_precision:.4f} - recall: {val_recall:.4f} - f1-score: {val_f1_score:.4f} - lr: 0.0010')

    return train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_labels = []
    all_predicted = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total

    # Calculate precision, recall, and f1-score
    precision = precision_score(all_labels, all_predicted, average='weighted')
    recall = recall_score(all_labels, all_predicted, average='weighted')
    f1 = f1_score(all_labels, all_predicted, average='weighted')

    return test_loss / len(test_loader), test_accuracy, precision, recall, f1

# Function to plot training history
def plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores):
    plt.figure(figsize=(12, 8))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title("Accuracy per Epoch")
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.title("Loss per Epoch")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # Precision plot
    plt.subplot(2, 2, 3)
    plt.plot(val_precisions, label='Validation Precision', color='green')
    plt.title("Precision per Epoch")
    plt.ylabel('Precision')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Recall and F1-Score plot
    plt.subplot(2, 2, 4)
    plt.plot(val_recalls, label='Validation Recall', color='purple')
    plt.plot(val_f1_scores, label='Validation F1-Score', color='red')
    plt.title("Recall and F1-Score per Epoch")
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

# Function to time the entire model training and evaluation process
def time_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    start_time = time.time()  # Start total timing

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)

    end_time = time.time()  # End total timing
    total_time = end_time - start_time
    print(f'Total training time: {total_time:.2f} seconds')

    return train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores

# Initialize data and model
if __name__ == "__main__":
    root = "./UNSW-NB15/"
    train_loader, test_loader, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(root)

    input_size = X_train_scaled.shape[1]  # Number of features
    hidden_size = 128  # Number of neurons in hidden layer

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DNN(input_size, hidden_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model and get results
    train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores = time_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=100)

    # Plot training history
    plot_training_history(train_losses, train_accuracies, val_losses, val_accuracies, val_precisions, val_recalls, val_f1_scores)
