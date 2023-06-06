import os
from matplotlib import pyplot as plt

import numpy as np
from gnn_utils import get_games_data
import utils.plot_utils as plot_utils
import data_cleaner
import data_reader
import utils.print_utils as print_utils
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import difflib
import random
import uuid
import math
from sklearn.model_selection import train_test_split


def plot_gnn_performance_metrics():
    # Read the performance metrics file
    file_path = 'trained_models/gnn2/gnn_performance_metrics.csv'
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return
    df = pd.read_csv(file_path)

    # Create the plots
    fig, axs = plt.subplots(1, figsize=(10, 10))
    # axs.plot(df['Train Loss'], label='Train Loss')
    # axs.plot(df['Val Loss'], label='Val Loss')
    # axs.set_xlabel('Iteracja')
    # axs.set_ylabel('Funkcja błędu (MSE)')
    # axs.legend()

    axs.plot(df['Train Accuracy'], label='Train Accuracy')
    axs.plot(df['Val Accuracy'], label='Val Accuracy')
    axs.set_xlabel('Iteracja')
    axs.set_ylabel('Precyzja')
    axs.legend()

    plt.show()

def plot_cnn_performance_metrics():
    # Read the performance metrics file
    file_path = 'trained_models/cnn/cnn_performance_metrics.csv'
    if not os.path.isfile(file_path):
        print(f"File {file_path} does not exist.")
        return
    df = pd.read_csv(file_path)

    # Create the plots
    fig, axs = plt.subplots(1, figsize=(10, 10))
    axs.plot(df['Train Loss'], label='Train Loss')
    axs.plot(df['Val Loss'], label='Val Loss')
    axs.set_xlabel('Iteracja')
    axs.set_ylabel('Funkcja błędu (MSE)')
    axs.legend()

    # axs.plot(df['Train Accuracy'], label='Train Accuracy')
    # axs.plot(df['Val Accuracy'], label='Val Accuracy')
    # axs.set_xlabel('Iteracja')
    # axs.set_ylabel('Precyzja')
    # axs.legend()

    plt.show()


if __name__ == '__main__':
    plot_cnn_performance_metrics()