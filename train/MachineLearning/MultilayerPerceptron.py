import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

# Định nghĩa MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size // 4, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return self.fc4(out)

# Hàm huấn luyện mô hình
def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        start_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        val_loss, val_accuracy, val_f1, val_precision, val_recall = evaluate_model(model, test_loader, criterion, device)

        train_losses.append(epoch_loss / len(train_loader))
        val_losses.append(val_loss)
        train_accuracies.append(100 * correct / total)
        val_accuracies.append(val_accuracy)

        end_time = time.time()
        epoch_time = end_time - start_time

        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'{len(train_loader)}/{len(train_loader)} [==============================] - '
              f'{epoch_time:.2f}s - loss: {epoch_loss/len(train_loader):.4e} - '
              f'accuracy: {100 * correct / total:.4f}% - '
              f'val_loss: {val_loss:.4e} - val_accuracy: {val_accuracy:.4f} - '
              f'val_f1: {val_f1:.4f} - val_precision: {val_precision:.4f} - val_recall: {val_recall:.4f}')

    return train_losses, val_losses, train_accuracies, val_accuracies

# Hàm đánh giá mô hình
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    all_labels = []
    all_predictions = []

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
            all_predictions.extend(predicted.cpu().numpy())

    test_accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')

    return test_loss / len(test_loader), test_accuracy, f1, precision, recall


# Khởi tạo dữ liệu và mô hình
def run_experiment(train_loader, test_loader, input_size, output_size, num_epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hidden_size = 64  # Giảm số neurons trong tầng ẩn
    model = MLP(input_size, hidden_size, output_size).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()

    train_losses, val_losses, train_accuracies, val_accuracies = train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed training time: {elapsed_time:.2f} seconds")

    test_loss, test_accuracy, test_f1, test_precision, test_recall = evaluate_model(model, test_loader, criterion, device)
    print(f'Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.2f}%, F1-score: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}')

    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies)

# Chuẩn bị dữ liệu và chạy mô hình
if __name__ == "__main__":
    root = "./UNSW-NB15/"
    train_loader, test_loader, X_train_scaled, X_test_scaled, y_train, y_test = load_and_preprocess_data(root)

    input_size = X_train_scaled.shape[1]
    output_size = len(set(y_train))

    run_experiment(train_loader, test_loader, input_size, output_size, num_epochs=100)
