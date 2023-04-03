import pandas as pd

def get_games_data(dataset):
    game_file = 'raw_data/' + dataset + '/item.csv'
    game_headers = ['game id', 'game title', 'Indie', 'Video Production',
                     'Action', 'Audio Production', 'Design & Illustration', 'Sports', 'Education',
                     'Web Publishing', 'Sexual Content', 'Gore', 'Racing', 'Violent', 'Adventure',
                     'Simulation', 'Strategy', 'Nudity', 'Early Access', 'RPG', 'Software Training',
                     'Casual', 'Game Development', 'Massively Multiplayer', 'Animation & Modeling',
                     'Utilities', 'Free to Play']
    game_df = pd.read_csv(game_file, sep=',', header=1,
                           names=game_headers, engine='python')
    return game_df

def get_game_name_by_id(df, id):
    return df.loc[df['game id'] == id]['game title'].values[0]

def get_game_names_by_id(df, id):
    return df.loc[df['game id'] == id]['game title'].tolist()