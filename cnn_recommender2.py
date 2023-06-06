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
from models.cnn_embeddings import ConvEmbeddingNet
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
lr = 1e-6
wd = 1e-4
bs = 400
n_epochs = 100
patience = 100
no_improvements = 0
best_loss = np.inf
best_weights = None
history = []
lr_history = []
use_scheduler = False
train_accuracy_list = []
val_accuracy_list = []
loss_through_epochs = []
val_loss_through_epochs = []
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
    global no_improvements
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
                    outputs = net(x_batch[:, 0], x_batch[:, 1], minmax)
                    loss = criterion(outputs, y_batch)
                    
                    # don't update weights and rates when in 'val' phase
                    if training:
                        if use_scheduler:
                            scheduler.step()
                        loss.backward()
                        optimizer.step()
                        if use_scheduler:
                            lr_history.extend(scheduler.get_lr())
                        
                running_loss += loss.item()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                loss_through_epochs.append(epoch_loss)
            stats[phase] = epoch_loss
            
            # early stopping: save weights of the best model so far
            if phase == 'val':
                val_loss_through_epochs.append(epoch_loss)
                if epoch_loss < best_loss:
                    print('loss improvement on epoch: %d' % (epoch + 1))
                    best_loss = epoch_loss
                    best_weights = copy.deepcopy(net.state_dict())
                    save_model(net, f"epoch_{epoch}_loss_{epoch_loss}")
                    save_metrics_to_csv()
                    no_improvements = 0
                else:
                    no_improvements += 1
                    save_model(net, f"epoch_{epoch}_loss_{epoch_loss}")
                    
        history.append(stats)
        print('[{epoch:03d}/{total:03d}] train: {train:.4f} - val: {val:.4f}'.format(**stats))
        if no_improvements >= patience:
            print('early stopping after epoch {epoch:03d}'.format(**stats))
            break


def test(model, datasets, dataset_sizes):
    model.eval()
    running_loss = 0.0
    train_correct = 0
    train_total = 0
    for batch in batches(*datasets['test'], bs=bs):
        x_batch, y_batch = [b.to(device) for b in batch]
        outputs = model(x_batch[:, 0], x_batch[:, 1])
        loss = criterion(outputs, y_batch)
        running_loss += loss.item()

        y_true = y_batch.round()
        train_pred_labels = outputs.round()
        train_correct += (train_pred_labels == y_true).sum().item()
        train_total += y_true.numel()
    

    train_accuracy = train_correct / train_total
    epoch_loss = running_loss / dataset_sizes['test']
    print(f"Test loss (MSE): {epoch_loss}, Test accuracy: {train_accuracy}")

def save_metrics_to_csv():
    # Create a dataframe to store the data
    metrics_df = pd.DataFrame({
        'Train Loss': loss_through_epochs,
        'Val Loss': val_loss_through_epochs,
        # 'Train Accuracy': train_accuracy_list,
        # 'Val Accuracy': val_accuracy_list
    })

    # Save the dataframe to a CSV file
    metrics_df.to_csv('cnn_performance_metrics.csv', index=False)

    print('Performance metrics saved to cnn_performance_metrics.csv.')

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
        torch.save(model.state_dict(), "trained_models/cnn2/cnnmodel.pt")
        return

    torch.save(model.state_dict(), f"trained_models/cnn2/{name}.pt")


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
    filename_train = f"data/steam/train_dataset.csv"
    filename_test = f"data/steam/test_dataset.csv"
    filename_val = f"data/steam/val_dataset.csv"

    dtypes = {
        'user_id': np.int32, 'game_id': np.int32,
        'rating': np.float32}
    data_train = pd.read_csv(
        filename_train, header=1,
        names=['user_id', 'game_id', 'rating'], dtype=dtypes)
 
    data_test = pd.read_csv(
        filename_test, header=1,
        names=['user_id', 'game_id', 'rating'], dtype=dtypes)
    
    data_val = pd.read_csv(
        filename_val, header=1,
        names=['user_id', 'game_id', 'rating'], dtype=dtypes)
 
    return (data_train, data_test, data_val)


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
    n_games = unique_games.shape[0]
    X = pd.DataFrame({'user_id': new_users, 'game_id': new_games})
    y = ratings['rating'].astype(np.float32)
    return (n_users, n_games), (X, y), (user_to_index, game_to_index)

def create_dataset2(ratings, top=None):
    if top is not None:
        ratings.groupby('user_id')['rating'].count()
 
    unique_users = ratings.user_id.unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)}
    new_users = ratings.user_id.map(user_to_index)
 
    unique_games = ratings.game_id.unique()
    game_to_index = {old: new for new, old in enumerate(unique_games)}
    new_games = ratings.game_id.map(game_to_index)
 
    X = pd.DataFrame({'user_id': new_users, 'game_id': new_games})
    y = ratings['rating'].astype(np.float32)
    return (X, y)

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

    train_df, test_df, val_df = get_datasets_dataframes()
    df = pd.concat([train_df, val_df])
    (n_users, n_games), (dataset, ratings), _ = create_dataset(df)
    train_dataset = dataset[:len(train_df)]
    train_ratings = ratings[:len(train_df)]
    val_dataset = dataset[len(train_df):len(train_df)+len(val_df)]
    val_ratings = ratings[len(train_df):len(train_df)+len(val_df)]
    test_dataset = dataset[len(train_df)+len(val_df):]
    test_ratings = ratings[len(train_df)+len(val_df):]

    # train_dataset, val_dataset, train_ratings, val_ratings = train_test_split(dataset, ratings, test_size=0.2, random_state=RANDOM_STATE)
    # train_dataset, train_ratings = create_dataset2(train_df)
    # val_dataset, val_ratings = create_dataset2(val_df)
    datasets = {'train': (train_dataset, train_ratings), 'val': (val_dataset, val_ratings)}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    minmax = df.rating.min().astype(float), df.rating.max().astype(float)
    # minmax = train_ratings.min(), train_ratings.max()
    # Instantiate the model with the appropriate number of features and categories
    net = EmbeddingNet(
        n_users=n_users, n_games=n_games,
        n_factors=1000, hidden=[200, 200],
        embedding_dropout=0.05, dropouts=[0.5, 0.5])
    
    # Load existing model weights
    if args.load_model or args.predict:
        load_existing_model(net, 'trained_models/cnn/epoch_73_loss_1.7103050363451975.pt')
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    iterations_per_epoch = int(math.ceil(dataset_sizes['train'] // bs))
    scheduler = CyclicLR(optimizer, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/10))
    if not args.load_model and not args.predict:
        train(net=net, datasets=datasets, optimizer=optimizer, minmax=minmax, scheduler=scheduler, dataset_sizes=dataset_sizes)

    if args.predict:
        (n_users, n_games), (test_dataset, test_ratings), _ = create_dataset(val_df)
        datasets = {'test': (test_dataset, test_ratings)}
        dataset_sizes = {'test': len(test_dataset)}
        print(dataset_sizes)
        test(net, datasets, dataset_sizes)

    # Save the model if specified and not loading existing model
    if args.save_model and not args.load_model:
        save_model(net, args.save_model_name)

    return


if __name__ == "__main__":
    run()


