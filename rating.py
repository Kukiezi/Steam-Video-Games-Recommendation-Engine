from typing import Any, NamedTuple
import torch
import torch.nn as nn
import pandas as pd
import argparse
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from training_types import CLIArguments, TrainingData
from recommendation_model import RecommendationModel
from utils.print_utils import print_testing_results 


NUM_EPOCHS = 1000


def parse_args() -> CLIArguments:
  parser = argparse.ArgumentParser()
  parser.add_argument('--load-model', dest="load_model", action='store_true', help='flag to specify whether to load a saved model')
  parser.add_argument('--save-model', dest="save_model", action='store_true', help='flag to specify whether to save a model')
  parser.add_argument('--save-test-data', dest="save_test_data", action='store_true', help='flag to specify whether to generate new split of data and save it to .csv files')
  return parser.parse_args()

def encode_categorical_columns(df):
  # Find the columns with categorical data
  categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
  # Encode the categorical columns
  le = LabelEncoder()
  for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
  return le.classes_


def get_features_and_label_columns(df):
  features = df.drop(columns=['rating'])
  label = df['rating']
  return (features, label)

def convert_to_tensors(data):
  data = data.to_numpy()
  data = torch.from_numpy(data)
  return data

def read_test_data_from_files() -> TrainingData:
  # Load features_train from a CSV file
  features_train = pd.read_csv('features_train.csv')

  # Load features_test from a CSV file
  features_test = pd.read_csv('features_test.csv')

  # Load label_train from a CSV file
  label_train = pd.read_csv('label_train.csv')

  # Load label_test from a CSV file
  label_test = pd.read_csv('label_test.csv')

  features_train = convert_to_tensors(features_train)
  features_test = convert_to_tensors(features_test)
  label_train = convert_to_tensors(label_train)
  label_test = convert_to_tensors(label_test)

  return TrainingData(features_train, features_test, label_train, label_test)


def split_data(features, label) -> TrainingData:
  # Split your data into training, evaluation, and test sets
  features_train, features_test, label_train, label_test = train_test_split(features, label, test_size=0.2)

  # Reshape the target tensors to have the same shape as the input tensors
 
  label_train = label_train.view(-1, 1)
  label_test = label_test.view(-1, 1)

  # Save features_train to a CSV file
  pd.DataFrame(features_train).to_csv('features_train.csv', index=False)

  # Save features_test to a CSV file
  pd.DataFrame(features_test).to_csv('features_test.csv', index=False)

  # Save label_train to a CSV file
  pd.DataFrame(label_train).to_csv('label_train.csv', index=False)

  # Save label_test to a CSV file
  pd.DataFrame(label_test).to_csv('label_test.csv', index=False)

  return TrainingData(features_train, features_test, label_train, label_test)

def train_model(model, features_train, label_train, loss_fn, optimizer):
  for epoch in range(NUM_EPOCHS):
    # Forward pass: compute predicted y by passing x to the model
    label_pred = model(features_train)
    # y_pred = torch.clamp(y_pred, 0, 10)

    # Ensure that y_pred and y_train have the same dtype
    label_pred = label_pred.to(label_train.dtype)
    label_train = label_train.to(label_pred.dtype)
    optimizer.zero_grad()

    # Print the ratings and predictions
    # print(f'Ratings: {y_train.numpy()}')
    # print(f'Predictions: {y_pred.detach().numpy()}')

    with torch.set_grad_enabled(True):
      # Compute and print loss
      loss = loss_fn(label_pred, label_train)

      print(f'Epoch {epoch + 1}: train loss = {loss.item():.4f}')

      # Zero gradients, perform a backward pass, and update the weights
      loss.backward()
      optimizer.step()

def test_model(model, features_test, label_test, loss_fn, game_title_mapping):
  with torch.no_grad():
    label_pred = model(features_test)
    print(features_test)
    # y_pred = torch.clamp(y_pred, 0, 10)

    test_loss = loss_fn(label_pred, label_test)

    # Print the testing results for each example
    for i in range(features_test.shape[0]):
      user_id = features_test[i, 0]
      game_title = game_title_mapping[int(features_test[i, 1])]
      rating = label_test[i, 0]
      predicted_rating = label_pred[i, 0]
      print_testing_results(user_id, game_title, rating, predicted_rating, test_loss)
    print(f'Test loss: {test_loss.item():.4f}')


def save_model(model):
  # Save the model
  torch.save(model.state_dict(), 'model.pt')

def load_existing_model(model, path):
  # Load existing model weights
  model.load_state_dict(torch.load(path))

def run():
  args = parse_args()

  # Load data into data_frame
  df = pd.read_csv('./data/cleaned_data2.csv', delimiter=',')
  game_title_categories  = encode_categorical_columns(df)
  game_title_mapping = {i: game_title_categories[i] for i in range(len(game_title_categories))}
  
  features, label = get_features_and_label_columns(df)

  # Instantiate the model with the appropriate number of features and categories
  model = RecommendationModel(num_features=features.shape[1], hidden_size=64)

  # Load existing model weights
  if args.load_model:
    load_existing_model(model, 'model.pt')

  # Define a loss function and an optimizer
  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  # Convert the data to tensors
  features = convert_to_tensors(features)
  label = convert_to_tensors(label)

  # Convert the input tensor to the same dtype as the weight tensor
  features = features.to(model.fc1.weight.dtype)

  # Split the data into training, evaluation, and test sets and assign them to invidual variables
  if args.save_test_data:
    features_train, features_test, label_train, label_test = split_data(features, label)
  else:
    features_train, features_test, label_train, label_test = read_test_data_from_files()
  features_train = features_train.to(model.fc1.weight.dtype)
  features_test = features_test.to(model.fc1.weight.dtype)

  iterations_per_epoch = int(math.ceil(len(features_train) // 2000))

  if not args.load_model:
    train_model(model, features_train, label_train, loss_fn, optimizer)

  # Test the model
  test_model(model, features_test, label_test, loss_fn, game_title_mapping)

  # Save the model
  if args.save_model:
    save_model(model)



if __name__ == '__main__':
  run()

  