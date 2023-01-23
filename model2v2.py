import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from itertools import zip_longest

class MF(nn.Module):
    def __init__(self, num_users: int, num_games: int, emb_size: int):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=emb_size)
        self.game_embedding = nn.Embedding(num_embeddings=num_games, embedding_dim=emb_size)
        self.user_embedding.weight.data.uniform_(0, 0.5)
        self.game_embedding.weight.data.uniform_(0, 0.5)

    def forward(self, user_id: torch.LongTensor, game_id: torch.LongTensor):
        user_embedding = self.user_embedding(user_id)
        game_embedding = self.game_embedding(game_id)
        dot_product = torch.sum(user_embedding * game_embedding, dim=1)
        dot_product = torch.sigmoid(dot_product)* 10
        # dot_product = torch.tanh(dot_product)
        return dot_product