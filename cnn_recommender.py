from typing import Any, NamedTuple
import torch
import torch.nn as nn
import pandas as pd
import argparse
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from gnn_utils import get_game_name_by_id, get_games_data
from models.model2 import MF
from custom_types.training_types import CLIArguments, TrainingDataMF as TrainingData
from utils.print_utils import print_testing_results
import numpy as np
import matplotlib.pyplot as plt  # plotting

# create lists to store the training loss, MAE, and RMSE for each epoch
loss_list = []
mae_list = []
rmse_list = []


def parse_args() -> CLIArguments:
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', dest="predict", action='store_true',
                        help='flag to specify whether to only do predictions')
    parser.add_argument('--load-model', dest="load_model", action='store_true',
                        help='flag to specify whether to load a saved model')
    parser.add_argument('--save-model', dest="save_model",
                        action='store_true', help='flag to specify whether to save a model')
    parser.add_argument('--save-model-name', dest="save_model_name",
                        help='flag to specify whether to save a model')
    parser.add_argument('--save-test-data', dest="save_test_data", action='store_true',
                        help='flag to specify whether to generate new split of data and save it to .csv files')
    return parser.parse_args()


def encode_categorical_columns(df):
    # Find the columns with categorical data
    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    # Encode the categorical columns
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])
    return le.classes_

# def train_epocs(model, train_df, loss_fn, epochs=50, lr=1e-3, wd=1e-5):


def train_epocs(model, train_df, loss_fn, optimizer, epochs=50):
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
    # for i, row in test_df.iterrows():
    #     user_id = row['user_id']
    #     game_title = row['game_title']
    #     rating = row['rating']
    #     predicted_rating = y_hat[i]
    #     print_testing_results(user_id, game_title, rating,
    #                           predicted_rating, loss)

    print("test loss %.3f " % loss.item())
    RMSE = np.sqrt(loss.item() / len(test_df))
    print("test RMSE %.3f " % RMSE)
    mae_loss = mae(y_hat, ratings)
    print("test MAE %.3f " % mae_loss.item())


def split_data(df) -> TrainingData:
    # Split your data into training and testing
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=44, shuffle=True)

    # resetting indices to avoid indexing errors
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return TrainingData(train_df, test_df)


def save_model(model, name):
    # Save the model
    if name is None:
        torch.save(model.state_dict(), "trained_models/cnnmodel.pt")
        return

    torch.save(model.state_dict(), f"trained_models/{name}.pt")


def load_existing_model(model, path):
    # Load existing model weights
    model.load_state_dict(torch.load(path))


def make_predictions(df, model):
    games_df = get_games_data('steam_200k')
    user = torch.tensor([107468])
    games = torch.tensor(df['game_id'].unique().tolist(), dtype=torch.long)
    predictions = model(user, games)

    sortedIndices = predictions.detach().numpy().argsort()

    top_game_ids = df['game_id'].unique(
    )[sortedIndices][:10]  # taking top 30
    
    top_game_names = []
    for game_id in top_game_ids:
        game_name = get_game_name_by_id(games_df, game_id)
        top_game_names.append(game_name)

    print(top_game_names)


def get_datasets_dataframes():
    filename_train = f"raw_data/steam_200k/train_dataset.csv"
    filename_test = f"raw_data/steam_200k/test_dataset.csv"
    dtypes = {
        'u_nodes': np.int32, 'v_nodes': np.int32,
        'ratings': np.float32}
    data_train = pd.read_csv(
        filename_train, header=1,
        names=['user_id', 'game_id', 'rating'], dtype=dtypes)

    data_test = pd.read_csv(
        filename_test, header=1,
        names=['user_id', 'game_id', 'rating'], dtype=dtypes)

    return (data_train, data_test)


def get_unique_counts(df1, df2):
    unique_user_count = pd.concat([df1['user_id'], df2['user_id']]).nunique()
    unique_game_count = pd.concat([df1['game_id'], df2['game_id']]).nunique()
    return unique_user_count, unique_game_count


def get_largest_user_id(df1, df2):
    largest_user_id = pd.concat([df1['user_id'], df2['user_id']]).max()
    return largest_user_id


def run():
    args = parse_args()

    df = pd.read_csv('./data/augmented-rounded.csv', delimiter=',')
    # game_title_categories  = encode_categorical_columns(df)
    # game_title_mapping = {i: game_title_categories[i] for i in range(len(game_title_categories))}
    # unique_user_count = df['user_id'].nunique()
    # unique_game_count = df['game_id'].nunique()
    # train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)

    # train_df, test_df = split_data(df)
    train_df, test_df = get_datasets_dataframes()
    unique_user_count, unique_game_count = get_unique_counts(train_df, test_df)

    # Instantiate the model with the appropriate number of features and categories
    model = MF(unique_user_count, unique_game_count, 100)

    # Load existing model weights
    if args.load_model or args.predict:
        load_existing_model(model, 'trained_models/cnn_100epochs.pt')

    if args.predict:
        make_predictions(df, model)
        return

    # Define a loss function and an optimizer
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=0.0001)

    if not args.load_model:
        train_epocs(model, loss_fn=loss_fn, train_df=train_df,
                    optimizer=optimizer, epochs=100)

    test(model, loss_fn=loss_fn, test_df=test_df)

    # Save the model if specified and not loading existing model
    if args.save_model and not args.load_model:
        save_model(model, args.save_model_name)

    # make_predictions(df, model)
    plt.plot(loss_list, label='loss')
    plt.plot(mae_list, label='MAE')
    plt.plot(rmse_list, label='RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run()
