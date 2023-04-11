# this file is trying to implement https://medium.com/coinmonks/how-to-implement-a-recommendation-system-with-deep-learning-and-pytorch-2d40476590f9

from typing import Any, NamedTuple
import torch
import torch.nn as nn
import pandas as pd
import argparse
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from cnn_reviews_iterator import ReviewsIterator
from cnn_utils import get_game_name_by_id, get_games_data
from models.cnn_embeddings import EmbeddingNet
from models.model2 import MF
from custom_types.training_types import CLIArguments, TrainingDataMF as TrainingData
from utils.print_utils import print_testing_results
import numpy as np
import matplotlib.pyplot as plt  # plotting
import copy
from torch.optim.lr_scheduler import _LRScheduler

class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


# create lists to store the training loss, MAE, and RMSE for each epoch
lr = 1e-3
wd = 1e-5
bs = 150
n_epochs = 100
patience = 10
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []


RANDOM_STATE = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.MSELoss(reduction='sum')

def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)


set_random_seed(RANDOM_STATE)


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


def train(net, datasets, optimizer, minmax, scheduler, dataset_sizes):
    global best_loss
    for epoch in range(n_epochs):
        stats = {'epoch': epoch + 1, 'total': n_epochs}
        
        for phase in ('train', 'val'):
            training = phase == 'train'
            running_loss = 0.0
            n_batches = 0
            for batch in batches(*datasets[phase], shuffle=training, bs=bs):
                x_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()
            
                # compute gradients only during 'train' phase
                with torch.set_grad_enabled(training):
                    outputs = net(x_batch[:, 1], x_batch[:, 0], minmax)
                    loss = criterion(outputs, y_batch)
                    
                    # don't update weights and rates when in 'val' phase
                    if training:
                        scheduler.step()
                        loss.backward()
                        optimizer.step()
                        lr_history.extend(scheduler.get_lr())
                        
                running_loss += loss.item()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            stats[phase] = epoch_loss
            
            # early stopping: save weights of the best model so far
            if phase == 'val':
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(net.state_dict())
                    save_model(net, 'net_model1.pt')
                    no_improvements = 0
                else:
                    no_improvements += 1
                    
        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break


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
    user = torch.tensor([100])
    # games = torch.tensor(df['game_id'].unique().tolist(), dtype=torch.long)
    games = torch.tensor([4000])
    predictions = model(user, games)

    print(predictions)
    print(predictions.detach().numpy())
    return
    sortedIndices = predictions.detach().numpy().argsort()
    print(sortedIndices)
    top_game_ids = df['game_id'].unique(
    )[[sortedIndices]][:10]  # taking top 30

    # top_game_names = []
    # for game_id in top_game_ids:
    #     game_name = get_game_name_by_id(games_df, game_id)
    #     top_game_names.append(game_name)

    print(top_game_ids)


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


def create_dataset(ratings, top=None):
    if top is not None:
        ratings.groupby('user_id')['rating'].count()

    unique_users = ratings.user_id.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.user_id.map(user_to_index)

    unique_games = ratings.game_id.unique()
    game_to_index = {old: new for new, old in enumerate(unique_games)}
    new_games = ratings.game_id.map(game_to_index)

    n_users = unique_users.shape[0]
    n_movies = unique_games.shape[0]
    X = pd.DataFrame({'user_id': new_users, 'game_id': new_games})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_movies), (X, y), (user_to_index, game_to_index)


def batches(X, y, bs=32, shuffle=True):
    for xb, yb in ReviewsIterator(X, y, bs, shuffle):
        xb = torch.LongTensor(xb)
        yb = torch.FloatTensor(yb)
        yield xb, yb.view(-1, 1)

def cosine(t_max, eta_min=0):
    
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min)*(1 + math.cos(math.pi*t/t_max))/2
    
    return scheduler

def run():
    args = parse_args()

    # df = pd.read_csv('./data/augmented-rounded.csv', delimiter=',')

    train_df, test_df = get_datasets_dataframes()
    games_df = get_games_data('steam_200k')
    df = pd.concat([train_df, test_df])
    (n, m), (X, y), _ = create_dataset(df)


    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    datasets = {'train': (X_train, y_train), 'val': (X_valid, y_valid)}
    dataset_sizes = {'train': len(X_train), 'val': len(X_valid)}

    minmax = df.rating.min().astype(float), df.rating.max().astype(float)
    
    # Instantiate the model with the appropriate number of features and categories
    net = EmbeddingNet(
        n_users=m, n_movies=n,
        n_factors=150, hidden=[500, 500, 500],
        embedding_dropout=0.05, dropouts=[0.5, 0.5, 0.25])
    
    # Load existing model weights
    if args.load_model or args.predict:
        load_existing_model(net, 'trained_models/net_model.pt')
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
    scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))
    if not args.load_model and not args.predict:
        train(net=net, datasets=datasets, optimizer=optimizer, minmax=minmax, scheduler=scheduler, dataset_sizes=dataset_sizes)

     
    if args.predict:
        make_predictions(df, net)

    # Save the model if specified and not loading existing model
    if args.save_model and not args.load_model:
        save_model(net, args.save_model_name)

    return


if __name__ == "__main__":
    run()


