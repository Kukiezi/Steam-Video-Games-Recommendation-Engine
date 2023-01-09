import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class RecommendationModel(nn.Module):
  def __init__(self, num_features, hidden_size):
    super(RecommendationModel, self).__init__()
    self.fc1 = nn.Linear(num_features, hidden_size)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)

    return x