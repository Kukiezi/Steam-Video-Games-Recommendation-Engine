import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from models.graphmodel2 import GCN

filename_train = f"raw_data/steam_200k/train_dataset.csv"
df = pd.read_csv(filename_train, names=['user_id', 'game_id', 'ratings'])
df = df.dropna()
df = df.astype(int)

users = df['user_id'].unique()
games = df['game_id'].unique()
nodes_dict = {}
for i, user in enumerate(users):
    nodes_dict[user] = i
for j, game in enumerate(games):
    nodes_dict[game] = j + len(users)


edges = []
for index, row in df.iterrows():
    source = nodes_dict[row['user_id']]
    target = nodes_dict[row['game_id']]
    rating = row['ratings']
    edges.append((source, target, rating))

edge_index = torch.tensor([edge[:2] for edge in edges], dtype=torch.long).t().contiguous()
edge_weight = torch.tensor([edge[2] for edge in edges], dtype=torch.float)
x = torch.zeros(len(nodes_dict), dtype=torch.float)
for node_id in nodes_dict.values():
    x[node_id] = 1
data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight)


train_mask = torch.zeros(len(nodes_dict), dtype=torch.bool)
train_mask[:len(users)] = 1
test_mask = ~train_mask
train_loader = DataLoader([data], batch_size=1)
model = GCN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

num_epochs = 50
for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, nodes_dict)
        loss = criterion(out[train_mask], batch.x[train_mask])
        loss.backward()
        optimizer.step()

    # Evaluate the model
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)
        mse_loss = criterion(out[test_mask], data.x[test_mask])
        print('Epoch: {:03d}, MSE Loss: {:.4f}'.format(epoch, mse_loss.item()))
        model.train()

