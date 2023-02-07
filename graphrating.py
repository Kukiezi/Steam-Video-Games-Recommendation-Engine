
from matplotlib import pyplot as plt    
import pandas as pd
from models.graphmodel import IGMC
import torch
from custom_types.training_types import TrainingDataMF as TrainingData
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from preprocessing import *
from util_functions import *
from data_utils import *

# Arguments
EPOCHS=2
BATCH_SIZE=50
LR=1e-3
LR_DECAY_STEP = 20
LR_DECAY_VALUE = 10

torch.manual_seed(123)
device = torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.synchronize()
    device = torch.device('cuda')

model = IGMC()
model.to(device)
model.reset_parameters()
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
df = pd.read_csv('./data/augmented-rounded.csv', delimiter=',', dtype={'steam_uid': str})

def split_data(df) -> TrainingData:
  # Split your data into training and testing
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)

  # resetting indices to avoid indexing errors
  train_df = train_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  return TrainingData(train_df, test_df)
R = pd.pivot_table(data=df, index='user_id', columns='game_title', values='rating')
(v_features, adj_train, train_labels, train_u_indices, train_v_indices, val_labels, 
val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, class_values
) = load_official_trainvaltest_split('steam_200k', testing=True)

train_dataset = eval('MyDynamicDataset')(root='data/ml_100k/testmode/train', A=adj_train, 
    links=(train_u_indices, train_v_indices), labels=train_labels, h=1, sample_ratio=1.0, 
    max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
test_dataset = eval('MyDataset')(root='data/ml_100k/testmode/test', A=adj_train, 
    links=(test_u_indices, test_v_indices), labels=test_labels, h=1, sample_ratio=1.0, 
    max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)

# def split_data(df) -> TrainingData:
#   # Split your data into training and testing
#   train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)

#   # resetting indices to avoid indexing errors
#   train_df = train_df.reset_index(drop=True)
#   test_df = test_df.reset_index(drop=True)

#   return TrainingData(train_df, test_df)

# def split_data(df) -> TrainingData:
#   # Split your data into training, evaluation, and test sets
#   train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)

#   # resetting indices to avoid indexing errors
#   train_df = train_df.reset_index(drop=True)
#   test_df = test_df.reset_index(drop=True)

# #   # Save train_df to a CSV file
# #   pd.DataFrame(train_df).to_csv('train_dataset.csv', index=False)

# #   # Save test_df to a CSV file
# #   pd.DataFrame(test_df).to_csv('test_dataset.csv', index=False)



#   return TrainingData(train_df, test_df)



# def create_plot_ranking_distribution():
#     fig = plt.figure()
#     ax = df_ratings.rating.value_counts(True).sort_index().plot.bar(figsize=(8,6))
#     vals = ax.get_yticks()
#     ax.yaxis.set_major_locator(plt.FixedLocator(vals))
#     ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
#     plt.xlabel('Rating', fontsize=12)
#     plt.ylabel('Share of Ratings', fontsize=12)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     fig.savefig('Ratings_distribution.png')

def train():
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss_all = 0
        for train_batch in train_loader:
            opt.zero_grad()
            train_batch = train_batch.to(device)
            y_pred = model(train_batch)
            y_true = train_batch.y
            train_loss = F.mse_loss(y_pred, y_true)
            train_loss.backward()
            train_loss_all += train_loss.item() * train_batch.num_graphs
            opt.step()
        train_loss_all /= len(train_loader.dataset)

    if epoch % LR_DECAY_STEP == 0:
        for param_group in opt.param_groups:
            param_group['lr'] = param_group['lr'] / LR_DECAY_VALUE
    print('epoch', epoch,'; train loss', train_loss_all)

def test():
    model.to(device)
    model.eval()
    test_loss = 0
    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        with torch.no_grad():
            y_pred = model(test_batch)
        y_true = test_batch.y
        test_loss += F.mse_loss(y_pred, y_true, reduction='sum')
    mse_loss = test_loss.item() / len(test_loader.dataset)
    print('test loss', mse_loss)

def main():
    
    # train_df, test_df = split_data(R)
    train()
    test()

if __name__ == "__main__":
    main()