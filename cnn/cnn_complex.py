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

class ASLCNN_V1(nn.Module):
    def __init__(self):
        super(ASLCNN_V1, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, padding=2)

        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(256 * 8 * 8, 1024)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class ASLCNN_V2(nn.Module):
    def __init__(self):
        super(ASLCNN_V2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.pool = nn.AvgPool2d(2)
        self.dropout1 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout2 = nn.Dropout(0.6)
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


class ASLCNN_V3(nn.Module):
    def __init__(self):
        super(ASLCNN_V3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ASLCNN_V4(nn.Module):
    def __init__(self):
        super(ASLCNN_V4, self).__init__()
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
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)

        x = x.view(-1, 128 * 8 * 8)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class ASLCNN_V5(nn.Module):
    def __init__(self):
        super(ASLCNN_V5, self).__init__()
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
        self.softmax = nn.Softmax(dim=1)

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
        x = self.softmax(x)

        return x
    

class ASLCNN_V6(nn.Module):
    def __init__(self):
        super(ASLCNN_V6, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)

        self.pool = nn.AvgPool2d(2)
        self.dropout1 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout2 = nn.Dropout(0.6)
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

class ASLCNN_V7(nn.Module):
    def __init__(self):
        super(ASLCNN_V7, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)

        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)

        self.pool = nn.AvgPool2d(2)
        self.dropout1 = nn.Dropout(0.35)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout2 = nn.Dropout(0.6)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    
# Training function
def train(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=10, momentum=0):
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

    # Training function
def train_F1(model, X_train, y_train, X_dev, y_dev, lr=1e-1, batch_size=32, num_epochs=10, momentum=0):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    best_dev_f1 =-1
    best_checkpoint = None
    best_epoch = -1

    for epoch in range(num_epochs):
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_func(logits, y_batch)
            loss.backward()
            optimizer.step()    
        train_acc, train_prec, train_rec, train_f1 = evaluate_F1(model, X_train, y_train, 'Train')
        dev_acc, dev_prec, dev_rec, dev_f1 = evaluate_F1(model, X_dev, y_dev, 'Dev')

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_checkpoint = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(f'Epoch {epoch}: Train F1: {train_f1:.4f}, Dev F1: {dev_f1:.4f}')

    model.load_state_dict(best_checkpoint)
    print(f'Best Epoch: {best_epoch}, Best Dev F1: {best_dev_f1:.4f}')
    

# Evaluation function
def evaluate(model, X, y, name):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_preds = torch.argmax(logits, dim=1)
        acc = torch.mean((y_preds == y).float()).item()
    print(f'{name} Accuracy: {acc:.5f}')
    return acc

# Evaluation with F1
def evaluate_F1(model, X, y, name):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        y_preds = torch.argmax(logits, dim=1)
        
        if name == "Test":
            incorrect_indices = torch.add((y_preds != y).nonzero(as_tuple=False), 1).squeeze()
            
            # Map each incorrect index into two values
            mapped_indices = [(index.item() // 150, index.item() % 150) for index in incorrect_indices]

            # Print out mapped indices of incorrect predictions
            print(f"Mapped incorrect predictions for {name}: {mapped_indices}")

        
        # macro averages
        precision = 0
        recall = 0
        f1 = 0
        for i in range(10):
            #p = (y_preds == i)
            #tp = torch.sum(torch.logical_and(t, p).int())
            #fp = torch.sum(torch.logical_and(~t, p).int())
            #fn = torch.sum(torch.logical_and(~t, ~p).int())
            tp = torch.sum(((y_preds == i) & (y == i)).to(torch.uint8))
            fp = torch.sum(((y_preds == i) & (y != i)).to(torch.uint8))
            fn = torch.sum(((y_preds != i) & (y == i)).to(torch.uint8))
            prec = tp / (tp+fp)
            precision += prec
            rec = tp/(tp+fn)
            recall += rec
            f1 += 2/((1/prec) + (1/rec))
            
        precision /= 10
        recall /= 10
        f1 /= 10

        acc = torch.mean((y_preds == y).float()).item()

    return (acc, precision, recall, f1)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-3)  # Adjusted default learning rate
    parser.add_argument('--batch-size', '-b', type=int, default=64)  # Adjusted default batch size
    parser.add_argument('--num-epochs', '-T', type=int, default=40)  # Adjusted default number of epochs
    parser.add_argument('--momentum', '-m', type=float, default=0.9)  # Adjusted default momentum
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(0)

    # Load the data
    all_data = np.load('asl_data_crop.npy', allow_pickle=True).item()
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
    model = ASLCNN_V5()

    # Train the model
    print("Training model...")
    train_F1(model, X_train, y_train, X_dev, y_dev, lr=args.learning_rate, 
          batch_size=args.batch_size, num_epochs=args.num_epochs, momentum=args.momentum)

    # Evaluate the model
    print("\nEvaluating model...")
    train_acc, train_prec, train_rec, train_f1 = evaluate_F1(model, X_train, y_train, 'Train')
    dev_acc, dev_prec, dev_rec, dev_f1 = evaluate_F1(model, X_dev, y_dev, 'Dev')
    test_acc, test_prec, test_rec, test_f1 = evaluate_F1(model, X_test, y_test, 'Test')
    
    print(f'Train Acc: {train_acc:.4f}, Dev Acc: {dev_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Train Prec: {train_prec:.4f}, Dev Prec: {dev_prec:.4f}, Test Prec: {test_prec:.4f}')
    print(f'Train Rec: {train_rec:.4f}, Dev Rec: {dev_rec:.4f}, Test Rec: {test_rec:.4f}')
    print(f'Train F1: {train_f1:.4f}, Dev F1: {dev_f1:.4f}, Test F1: {test_f1:.4f}')

    print ()
if __name__ == '__main__':
    main()
