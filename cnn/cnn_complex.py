import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import copy
import time
import argparse

# Define the ASLCNN model
class ASLCNN(nn.Module):
    def __init__(self):
        super(ASLCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Training function
def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=30, momentum=0):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_dev_acc = -1
    best_checkpoint = None
    best_epoch = -1

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_func(logits, y_batch)
            loss.backward()
            optimizer.step()

        train_acc = evaluate(model, X_train, y_train, 'Train')
        dev_acc = evaluate(model, X_dev, y_dev, 'Dev')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_checkpoint = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(f'Epoch {epoch}: Train Accuracy: {train_acc:.4f}, Dev Accuracy: {dev_acc:.4f}')

    model.load_state_dict(best_checkpoint)
    print(f'Best Epoch: {best_epoch}, Best Dev Accuracy: {best_dev_acc:.4f}')

# Evaluation function
def evaluate(model, X, y, name):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_preds = torch.argmax(logits, dim=1)
        acc = torch.mean((y_preds == y).float()).item()
    print(f'{name} Accuracy: {acc:.5f}')
    return acc

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-3)  # Adjusted default learning rate
    parser.add_argument('--batch-size', '-b', type=int, default=64)  # Adjusted default batch size
    parser.add_argument('--num-epochs', '-T', type=int, default=20)  # Adjusted default number of epochs
    parser.add_argument('--momentum', '-m', type=float, default=0.9)  # Adjusted default momentum
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(0)

    # Load the data
    all_data = np.load('asl_data.npy', allow_pickle=True).item()
    X_train = torch.tensor(all_data['X_train'], dtype=torch.float)
    y_train = torch.tensor(all_data['y_train'], dtype=torch.long)
    X_dev = torch.tensor(all_data['X_dev'], dtype=torch.float)
    y_dev = torch.tensor(all_data['y_dev'], dtype=torch.long)
    X_test = torch.tensor(all_data['X_test'], dtype=torch.float)
    y_test = torch.tensor(all_data['y_test'], dtype=torch.long)

    # Reshape the input data
    X_train = X_train.view(-1, 1, 64, 64)  # Reshape to (N, C, H, W) format
    X_dev = X_dev.view(-1, 1, 64, 64)
    X_test = X_test.view(-1, 1, 64, 64)

    # Create the ASLCNN model
    model = ASLCNN()

    # Train the model
    print("Training model...")
    train(model, X_train, y_train, X_dev, y_dev, lr=args.learning_rate, 
          batch_size=args.batch_size, num_epochs=args.num_epochs, momentum=args.momentum)

    # Evaluate the model
    print("\nEvaluating model...")
    train_acc = evaluate(model, X_train, y_train, 'Train')
    dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
    test_acc = evaluate(model, X_test, y_test, 'Test')

    print(f'\nFinal Performance: Train Acc: {train_acc:.5f}, Dev Acc: {dev_acc:.5f}, Test Acc: {test_acc:.5f}')

if __name__ == '__main__':
    main()
