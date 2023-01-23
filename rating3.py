from typing import Any, NamedTuple
import torch
import torch.nn as nn
import pandas as pd
import argparse
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model2 import MF
from model3 import EmbeddingNet
from training_types import CLIArguments, TrainingData
from recommendation_model import RecommendationModel
from utils.print_utils import print_testing_results 
import numpy as np

lr = 1e-3
wd = 1e-5
bs = 2000
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []
# use GPU if available
identifier = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(identifier)

def encode_categorical_columns(df):
  # Find the columns with categorical data
  categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
  # Encode the categorical columns
  le = LabelEncoder()
  for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
  return le.classes_


def train_epocs(model, train_df, loss_fn, optimizer, epochs=50, lr=0.01, wd=0.0):
    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            usernames = torch.LongTensor(train_df.user_id.values)
            game_titles = torch.LongTensor(train_df.game_id.values)
            ratings = torch.FloatTensor(train_df.rating.values)
            outputs = model(usernames, game_titles)
            loss = loss_fn(outputs, ratings)
            loss.backward()
            optimizer.step()
        print(f"{i}: {loss.item()}")
        RMSE = np.sqrt(loss.item() / len(train_df))
        print("train RMSE %.3f " % RMSE)


def test(model, test_df, loss_fn):
      model.eval()
      usernames = torch.LongTensor(test_df.user_id.values)
      game_titles = torch.LongTensor(test_df.game_id.values)
      ratings = torch.FloatTensor(test_df.rating.values)
      y_hat = model(usernames, game_titles)
      loss = loss_fn(y_hat, ratings)
      # for i,row in test_df.iterrows():
      # 	user_id = row['user_id']
      # 	game_title = row['game_title']
      # 	rating = row['rating']
      # 	predicted_rating = y_hat[i]
      # 	print_testing_results(user_id, game_title, rating, predicted_rating, loss)
      
      print("test loss %.3f " % loss.item())
      RMSE = np.sqrt(loss.item() / len(test_df))
      print("test RMSE %.3f " % RMSE)

def make_predictions(df, model):
  user = torch.tensor([2])
  games = torch.tensor(df['game_id'].unique().tolist())
  predictions = model(user, games)

  sortedIndices = predictions.detach().numpy().argsort()

  recommendations = df['game_title'].unique()[sortedIndices][:10]  # taking top 30
  print(recommendations)

def main():
    df = pd.read_csv('./data/cdata3.csv', delimiter=',')
    # game_title_categories  = encode_categorical_columns(df)
    # game_title_mapping = {i: game_title_categories[i] for i in range(len(game_title_categories))}
    unique_user_count = df['user_id'].nunique()
    unique_game_count = df['game_id'].nunique()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)
    # resetting indices to avoid indexing errors
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    loss_fn = nn.MSELoss(reduction='sum')
    model = EmbeddingNet(unique_user_count, unique_game_count)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # iterations_per_epoch = int(math.ceil(train_df.shape[0] // bs))
    print(unique_user_count)
    print(unique_game_count)
    

    train_epocs(model, loss_fn=loss_fn, train_df=train_df, optimizer=optimizer, epochs=100)
    test(model, loss_fn=loss_fn, test_df=test_df)

    make_predictions(df, model)
if __name__ == "__main__":
    main()