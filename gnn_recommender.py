
from matplotlib import pyplot as plt    
import pandas as pd
from gnn_utils import get_game_name_by_id, get_games_data
from models.graphmodel import IGMC
import torch
from custom_types.training_types import TrainingDataMF as TrainingData
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from custom_dataset import CustomDataset
from preprocessing import *
from util_functions import *
from data_utils import *
import torch.nn.functional as F
import seaborn as sns

# Arguments
EPOCHS=10
BATCH_SIZE=50
LR=1e-3
LR_DECAY_STEP = 20
LR_DECAY_VALUE = 10


# df = pd.read_csv('./data/augmented-rounded.csv', delimiter=',', dtype={'steam_uid': str})

def split_data(df) -> TrainingData:
  # Split your data into training and testing
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)

  # resetting indices to avoid indexing errors
  train_df = train_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  return TrainingData(train_df, test_df)

def train(model, opt, device, train_loader):
    loss_through_epochs = []
    batches_per_epoch = len(train_loader)
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss_all = 0
        for i, train_batch in enumerate(train_loader):
            print(f"{i}/{batches_per_epoch}")
            opt.zero_grad()
            train_batch = train_batch.to(device)
            y_pred = model(train_batch)
            y_true = train_batch.y
            train_loss = F.mse_loss(y_pred, y_true)
            train_loss.backward()
            train_loss_all += BATCH_SIZE * float(train_loss)
            opt.step()
            torch.cuda.empty_cache()
    train_loss_all = train_loss_all / len(train_loader.dataset)
    loss_through_epochs.append(train_loss_all)
    print('epoch', epoch,'; train loss', train_loss_all)

    if epoch % LR_DECAY_STEP == 0:
      for param_group in opt.param_groups:
          param_group['lr'] = param_group['lr'] / LR_DECAY_VALUE
          
def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    y_real=[]
    pred=[]
    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        with torch.no_grad():
            y_pred = model(test_batch)
            y_true = test_batch.y
            test_loss += F.mse_loss(y_pred, y_true, reduction='sum')

            y_real += test_batch.y.tolist()
            pred += y_pred.tolist()
    df = pd.DataFrame()
    df["y_real"] = y_real
    df["y_pred"] = pred
    df["y_real"] = df["y_real"].astype(float)
    df["y_pred"] = df["y_pred"].astype(float)
    # Print each prediction in the required format
    for i in range(len(y_real)):
        print(f"Predicted rating: {pred[i]}, actual rating: {y_real[i]}")

    mse_loss = test_loss.item() / len(test_loader.dataset)
    print('test loss', mse_loss)


def recommend_games(user_id, model, device, test_dataset, k=10):
    # Get the list of all game_ids
    game_ids = np.unique(test_dataset.data['edge_index'][1])
    
    # Create a mask to filter out the games the user has already played
    mask = (test_dataset.data['edge_index'][0] == user_id)
    played_games = test_dataset.data['edge_index'][1][mask]
    not_played_mask = np.isin(game_ids, played_games, invert=True)
    not_played_games = game_ids[not_played_mask]
    
    # Create a tensor of user ids and not played game ids
    user_ids = torch.full((not_played_games.size,), user_id, dtype=torch.long, device=device)
    game_ids_tensor = torch.tensor(not_played_games, dtype=torch.long, device=device)
    test_data = torch.stack([user_ids, game_ids_tensor], dim=0)

    # Predict the ratings for the user and not played games
    model.eval()
    with torch.no_grad():
        predictions = model(test_data)

    # Sort the not played games by predicted rating
    sorted_games = not_played_games[torch.argsort(predictions.view(-1), descending=True)]

    # Return the top k recommended games
    return sorted_games[:k]

def recommend_games_for_user(user_id, model, device, game_ids):
    edges = []
    neutral_rating = -1  # or use the neutral rating from your dataset

    # Create a list of edges for all games with the given user ID
    for game_id in game_ids:
        edges.append((user_id, game_id, neutral_rating))

    # Convert the list of tuples to a PyTorch tensor
    edges = torch.tensor(edges, dtype=torch.long).to(device)
    with torch.no_grad():
        predicted_ratings = model(edges)

    # Sort the predicted ratings in descending order and select the top 10 game IDs
    top_10_indices = torch.argsort(predicted_ratings, descending=True)[:10]
    top_10_game_ids = [game_ids[i] for i in top_10_indices]

    return top_10_game_ids


def get_all_game_ids():
    # Load train and test datasets
    train_dataset = f"raw_data/steam_200k/train_dataset.csv"
    test_dataset= f"raw_data/steam_200k/test_dataset.csv"
    train_df = pd.read_csv(train_dataset)
    test_df = pd.read_csv(test_dataset)
    # Concatenate the two dataframes
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    game_ids = all_df['game_id'].unique().tolist()
    return game_ids

def get_user_game_tuples(users, games):
    user_game_tuples = []
    neutral_rating = -1
    for user_id in users:
        for game_id in games:
            user_game_tuples.append((user_id, game_id, neutral_rating))
    return user_game_tuples

games_to_predict = [39, 45, 61, 221, 53, 604]

def get_predictions(model, dataloader, movies_to_predict):
    predictions = dict()
    batches_for_prediction = len(games_to_predict)
    device = torch.device("cpu")
    for i, test_batch in enumerate(dataloader):
        print(f"{i}/{batches_for_prediction - 1}")
        test_batch = test_batch.to(device)
        with torch.no_grad():
            y_my_pred = model(test_batch)
        torch.cuda.empty_cache()
        predictions[movies_to_predict[i]] = y_my_pred[-1].item()
        print(predictions)
        if i == batches_for_prediction - 1:
            break
    return predictions

def print_best_N_predictions(predictions, movies, n=10):
    i = 0
    d_view = [(v, k) for k, v in predictions.items()]
    d_view.sort(reverse=True)
    print(d_view)
    print(f"Best {n} predictions by possible review:")
    for v, k in d_view:
        if i == n:
            break
        print(f"idx {k}; {get_game_name_by_id(movies, k)}: {v}")
        i += 1

def main():
    torch.manual_seed(123)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.synchronize()
        device = torch.device('cuda')

    model = IGMC()
    # model.to(device)
    # model.reset_parameters()

    model.load_state_dict(torch.load("graph_10_epochs.pt"))
   

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0)
    (v_features, adj_train, train_labels, train_u_indices, train_v_indices, val_labels, 
    val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, class_values
    ) = load_official_trainvaltest_split('steam_200k', testing=True)

    # train_dataset = eval('MyDynamicDataset')(root='data/ml_100k/testmode/train', A=adj_train, 
    #     links=(train_u_indices, train_v_indices), labels=train_labels, h=1, sample_ratio=1.0, 
    #     max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)

    # test_dataset = eval('MyDataset')(root='data/ml_100k/testmode/test', A=adj_train, 
    #     links=(test_u_indices, test_v_indices), labels=test_labels, h=1, sample_ratio=1.0, 
    #     max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
    
    test_dataset = eval('MyDataset')(root='data/ml_100k/processed/test', A=adj_train, 
        links=(test_u_indices, test_v_indices), labels=test_labels, h=1, sample_ratio=1.0, 
        max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
    
    # train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)
    # train_df, test_df = split_data(R)
    # train(model, opt, device, train_loader)
    # torch.save(model.state_dict(), "graph_10_epochs.pt")
    # print('Model saved!')
    test(model, device, test_loader)
    # predictions = get_predictions(model, test_loader, games_to_predict)
    all_games = get_games_data('steam_200k')
    # print(get_game_name_by_id(all_games, max(predictions, key=predictions.get)))
    # print_best_N_predictions(predictions, all_games, n=10)

if __name__ == "__main__":
    main()


def make_recommendations():
    ratings_df = pd.DataFrame({'user_id': [12345689, 123456789, 123456789, 123456789, 123456789], 
                           'game_id': [39, 45, 61, 221, 53], 
                           'ratings': [9, 10, 7, 2, 10]})
