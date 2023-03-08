import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class GameRatingsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user_id, game_id, rating = self.data[index]
        return torch.tensor([user_id, game_id]), torch.tensor(rating)
    
class GameRatingsModel(nn.Module):
    def __init__(self, num_users, num_games, embedding_size, hidden_size):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.game_embedding = nn.Embedding(num_games, embedding_size)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=2)
        self.fc1 = nn.Linear(hidden_size * 31, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        user_ids = inputs[:, 0]
        game_ids = inputs[:, 1]

        user_embedded = self.user_embedding(user_ids)
        game_embedded = self.game_embedding(game_ids)

        x = torch.cat((user_embedded.unsqueeze(2), game_embedded.unsqueeze(2)), dim=2)
        x = x.unsqueeze(1)

        x = self.conv1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.fc2(x)

        return x

def train(model, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, targets.float())
            total_loss += loss.item() * len(inputs)
            total_items += len(inputs)
    return total_loss / total_items

# Load data
filename_train = f"raw_data/steam_200k/train_dataset.csv"
data = np.loadtxt(filename_train, delimiter=",", dtype=int)
train_data, val_data = train_test_split(data, test_size=0.2)
train_dataset = GameRatingsDataset(train_data)
val_dataset = GameRatingsDataset(val_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define model and hyperparameters
num_users = max(data[:, 0]) + 1
num_games = max(data[:, 1]) + 1
embedding_size = 32
hidden_size = 64
model = GameRatingsModel(num_users, num_games, embedding_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train model
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Predict ratings
filename_test = f"raw_data/steam_200k/test_dataset.csv"
test_data = np.loadtxt(filename_test, delimiter=",", dtype=int)
# test_data = pd.read_csv(filename_test, names=['user_id', 'game_id', 'ratings'])
test_dataset = GameRatingsDataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=64)
model.eval()
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs).squeeze()
        for i in range(len(outputs)):
            user_id, game_id = inputs[i]
            predicted_rating = outputs[i]
            actual_rating = targets[i]
            print(f"User ID: {user_id}, Game ID: {game_id}, Predicted Rating: {predicted_rating:.2f}, Actual Rating: {actual_rating}")
        # Do something with the predictions (e.g. save to file)
