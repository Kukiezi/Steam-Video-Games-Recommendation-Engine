from typing import Any, NamedTuple
import torch
import torch.nn as nn
import pandas as pd
import argparse
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model2 import MF
from training_types import CLIArguments, TrainingData
from recommendation_model import RecommendationModel
from utils.print_utils import print_testing_results 
import numpy as np
import matplotlib.pyplot as plt  # plotting

# create lists to store the training loss, MAE, and RMSE for each epoch
loss_list = []
mae_list = []
rmse_list = []

def encode_categorical_columns(df):
  # Find the columns with categorical data
  categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
  # Encode the categorical columns
  le = LabelEncoder()
  for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
  return le.classes_


def train_epocs(model, train_df, loss_fn, epochs=50, lr=0.01, wd=0.0001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    mae = nn.L1Loss()
    for i in range(epochs):
        usernames = torch.LongTensor(train_df.user_id.values)
        game_titles = torch.LongTensor(train_df.game_id.values)
        ratings = torch.FloatTensor(train_df.rating.values)
        y_hat = model(usernames, game_titles)
        loss = loss_fn(y_hat, ratings)
        optimizer.zero_grad()  # reset gradient
        loss.backward()
        optimizer.step()
        print(f"{i}: {loss.item()}")
        RMSE = np.sqrt(loss.item() / len(train_df))
        print("train RMSE %.3f " % RMSE)
        mae_loss = mae(y_hat, ratings)
        print("test MAE %.3f " % mae_loss.item())
        loss_list.append(loss.item())
        mae_list.append(mae_loss.item())
        rmse_list.append(RMSE)
        


def test(model, test_df, loss_fn):
      model.eval()
      usernames = torch.LongTensor(test_df.user_id.values)
      game_titles = torch.LongTensor(test_df.game_id.values)
      ratings = torch.FloatTensor(test_df.rating.values)
      y_hat = model(usernames, game_titles)
      loss = loss_fn(y_hat, ratings)
      mae = nn.L1Loss()
      # for i,row in test_df.iterrows():
      #     user_id = row['user_id']
      #     game_title = row['game_title']
      #     rating = row['rating']
      #     predicted_rating = y_hat[i]
      #     print_testing_results(user_id, game_title, rating, predicted_rating, loss)
      
      print("test loss %.3f " % loss.item())
      RMSE = np.sqrt(loss.item() / len(test_df))
      print("test RMSE %.3f " % RMSE)
      mae_loss = mae(y_hat, ratings)
      print("test MAE %.3f " % mae_loss.item())

def make_predictions(df, model):
  user = torch.tensor([2])
  games = torch.tensor(df['game_id'].unique().tolist())
  predictions = model(user, games)

  sortedIndices = predictions.detach().numpy().argsort()

  recommendations = df['game_title'].unique()[sortedIndices][:10]  # taking top 30
  print(recommendations)

def main():
    df = pd.read_csv('./data/augmented.csv', delimiter=',')
    # game_title_categories  = encode_categorical_columns(df)
    # game_title_mapping = {i: game_title_categories[i] for i in range(len(game_title_categories))}
    unique_user_count = df['user_id'].nunique()
    unique_game_count = df['game_id'].nunique()
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)
    # resetting indices to avoid indexing errors
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    loss_fn = nn.MSELoss()

    print(unique_user_count)
    print(unique_game_count)
    model = MF(unique_user_count, unique_game_count, 100)
    train_epocs(model, loss_fn=loss_fn, train_df=train_df, epochs=10)
    test(model, loss_fn=loss_fn, test_df=test_df)

    make_predictions(df, model)
    plt.plot(loss_list, label='loss')
    plt.plot(mae_list, label='MAE')
    plt.plot(rmse_list, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()