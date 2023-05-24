import csv
from matplotlib import pyplot as plt    
import pandas as pd
from gnn_utils import get_game_name_by_id, get_games_data
from models.graphmodel import IGMC
import torch
from custom_types.training_types import CLIArguments, TrainingDataMF as TrainingData
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from custom_dataset import CustomDataset
from preprocessing import *
from util_functions import *
from data_utils import *
import torch.nn.functional as F
import copy


# Arguments
best_loss = np.inf
EPOCHS=100
BATCH_SIZE=50
LR=1e-3
LR_DECAY_STEP = 20
WD = 1e-5
LR_DECAY_VALUE = 10
train_accuracy_list = []
train_val_accuracy_list = []
val_accuracy_list = []
loss_through_epochs = []
val_loss_through_epochs = []
# df = pd.read_csv('./data/augmented-rounded.csv', delimiter=',', dtype={'steam_uid': str})

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


def compute_accuracy(predictions, targets):
    # Assuming predictions and targets are tensors or numpy arrays
    correct = (predictions == targets).sum().item()
    total = predictions.numel()
    accuracy = correct / total
    return accuracy

def split_data(df) -> TrainingData:
  # Split your data into training and testing
  train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)

  # resetting indices to avoid indexing errors
  train_df = train_df.reset_index(drop=True)
  test_df = test_df.reset_index(drop=True)

  return TrainingData(train_df, test_df)

def train(model, opt, device, train_loader, validation_loader):
    global best_loss
    batches_per_epoch = len(train_loader)
    for epoch in range(1, EPOCHS+1):
        model.train()
        train_loss_all = 0
        train_correct = 0
        train_total = 0
        for i, train_batch in enumerate(train_loader):
            if i % 1000 == 0:
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
            # Compute accuracy for training data
            train_pred_labels = y_pred.round().long()
            train_correct += (train_pred_labels == y_true).sum().item()
            train_total += y_true.numel()

        train_loss_all = train_loss_all / len(train_loader.dataset)
        loss_through_epochs.append(train_loss_all)
        train_accuracy = train_correct / train_total
        train_accuracy_list.append(train_accuracy)
        print('epoch', epoch, '; train loss', train_loss_all, '; train accuracy', train_accuracy)

        # Model Validation
        model.eval()
        val_loss = 0
        y_real=[]
        pred=[]
        val_total = 0
        val_correct = 0
        for i, val_batch in enumerate(validation_loader):
            val_batch = val_batch.to(device)
            val_predictions = []
            with torch.no_grad():
                y_pred = model(val_batch)
                y_true = val_batch.y
                val_loss += F.mse_loss(y_pred, y_true, reduction='sum')

                y_real += val_batch.y.tolist()
                pred += y_pred.tolist()
                # Compute accuracy for validation data
                train_pred_labels = y_pred.round().long()
                val_correct += (train_pred_labels == y_true).sum().item()
                val_total += y_true.numel()


        mse_loss = val_loss.item() / len(validation_loader.dataset)
        val_accuracy = val_correct / val_total

        val_accuracy_list.append(val_accuracy)
        print('validation loss', mse_loss, '; validation accuracy', val_accuracy)
        val_loss_through_epochs.append(mse_loss)

        # save model if it has the best validation accuracy
        if mse_loss < best_loss:
            print('loss improvement on epoch: %d' % (epoch + 1))
            best_loss = mse_loss
            best_weights = copy.deepcopy(model.state_dict())
            save_model(model, f"epoch_{epoch}_loss_{mse_loss}")
            save_metrics_to_csv()
            no_improvements = 0
        else:
            no_improvements += 1
            if no_improvements >= 10:
                print('no improvements for 10 epochs, stopping training')
                break

        if epoch % LR_DECAY_STEP == 0:
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr'] / LR_DECAY_VALUE
          
def test(model, device, test_loader):
    model.to(device)
    model.eval()
    test_loss = 0
    y_real=[]
    pred=[]
    train_correct = 0
    train_total = 0
    for test_batch in test_loader:
        test_batch = test_batch.to(device)
        with torch.no_grad():
            y_pred = model(test_batch)
            y_true = test_batch.y
            test_loss += F.mse_loss(y_pred, y_true, reduction='sum')

            y_real += test_batch.y.tolist()
            pred += y_pred.tolist()

            train_pred_labels = y_pred.round().long()
            train_correct += (train_pred_labels == y_true.round().long()).sum().item()
            train_total += y_true.numel()
    df = pd.DataFrame()
    df["y_real"] = y_real
    df["y_pred"] = pred
    df["y_real"] = df["y_real"].astype(float)
    df["y_pred"] = df["y_pred"].astype(float)
    # Print each prediction in the required format
    # for i in range(len(y_real)):
    #     print(f"Predicted rating: {pred[i]}, actual rating: {y_real[i]}")

    mse_loss = test_loss.item() / len(test_loader.dataset)
    # print('test loss', mse_loss)
    train_accuracy = train_correct / train_total
    train_accuracy_list.append(train_accuracy)
    print('test loss', mse_loss, '; test accuracy', train_accuracy)


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

def save_metrics_to_csv():
    # Create a dataframe to store the data
    metrics_df = pd.DataFrame({
        'Train Loss': loss_through_epochs,
        'Val Loss': val_loss_through_epochs,
        'Train Accuracy': train_accuracy_list,
        'Val Accuracy': val_accuracy_list
    })

    # Save the dataframe to a CSV file
    metrics_df.to_csv('gnn_performance_metrics.csv', index=False)

    print('Performance metrics saved to gnn_performance_metrics.csv.')


def save_model(model, name):
    # Save the model
    if name is None:
        torch.save(model.state_dict(), "trained_models/gnn2/gnn_model.pt")
        return

    torch.save(model.state_dict(), f"trained_models/gnn2/{name}.pt")

def main():
    args = parse_args()


    torch.manual_seed(123)
    device = torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        torch.cuda.synchronize()
        device = torch.device('cuda')

    model = IGMC()

    # Load existing model weights
    # if args.load_model or args.predict:
        # model.load_state_dict(torch.load("trained_models/gnn/epoch_7_loss_1.072769907117512.pt.pt"))

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    (v_features, adj_train, train_labels, train_u_indices, train_v_indices, val_labels, 
    val_u_indices, val_v_indices, test_labels, test_u_indices, test_v_indices, class_values
    ) = load_official_trainvaltest_split('steam_200k', testing=True)

    train_dataset = eval('MyDynamicDataset')(root='data/gnn_steam/testmode/train', A=adj_train, 
        links=(train_u_indices, train_v_indices), labels=train_labels, h=1, sample_ratio=1.0, 
        max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)

    test_dataset = eval('MyDataset')(root='data/gnn_steam/testmode/test', A=adj_train, 
        links=(test_u_indices, test_v_indices), labels=test_labels, h=1, sample_ratio=1.0, 
        max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
    
    validation_dataset = eval('MyDataset')(root='data/gnn_steam/testmode/validate', A=adj_train, 
        links=(val_u_indices, val_v_indices), labels=val_labels, h=1, sample_ratio=1.0, 
        max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
    # test_dataset = eval('MyDataset')(root='data/ml_100k/processed/test', A=adj_train, 
    #     links=(test_u_indices, test_v_indices), labels=test_labels, h=1, sample_ratio=1.0, 
    #     max_nodes_per_hop=200, u_features=None, v_features=None, class_values=class_values)
    
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=2)

    validation_loader = DataLoader(validation_dataset, BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=2)
    # train_df, test_df = split_data(R)
    if not args.load_model and not args.predict:
        train(model, opt, device, train_loader, validation_loader)
    # torch.save(model.state_dict(), "graph_10_epochs.pt")
    # print('Model saved!')
    if args.predict:
        test(model, device, test_loader)

    # Save the model if specified and not loading existing model
    # if args.save_model and not args.load_model:
    #     save_model(model, args.save_model_name)
    # predictions = get_predictions(model, test_loader, games_to_predict)
    # all_games = get_games_data('steam_200k')
    # print(get_game_name_by_id(all_games, max(predictions, key=predictions.get)))
    # print_best_N_predictions(predictions, all_games, n=10)

if __name__ == "__main__":
    main()


def make_recommendations():
    ratings_df = pd.DataFrame({'user_id': [12345689, 123456789, 123456789, 123456789, 123456789], 
                           'game_id': [39, 45, 61, 221, 53], 
                           'ratings': [9, 10, 7, 2, 10]})
