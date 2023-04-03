import os
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
from sklearn.model_selection import train_test_split

# plot_utils.plotPerColsumnDistribution(df1, 10, 5)


def main():
    # data_frame = pd.read_csv('./data/cleaned_data.csv', delimiter=',', nrows=None)
    # print(data_frame.head(5))
    steam_200k = data_cleaner.clean_data(data_reader.read_data_into_data_frame('./data/data.csv', 'steam_200k'))
    print_utils.print_row_and_col_count(steam_200k)
    # print(data_frame['game_title'].value_counts())
    steam = data_cleaner.clean_data(data_reader.read_data_into_data_frame('./data/steam.csv', 'steam'))
    print_utils.print_row_and_col_count(steam)

    steam_200k = steam_200k.loc[steam_200k['behavior'] != 'purchase']
    
    # merge
    # data_frame = steam_200k.merge(steam, left_on='game_title', right_on='name')
    data_frame = pd.DataFrame(columns=steam_200k.columns.tolist() + steam.columns.tolist())
    # Iterate over the rows of steam_200k
    for index, row in steam_200k.iterrows():
        # Get the game_title of the current row
        game_title = row['game_title']
        print(index)
        # Find the row in steam with a name that is most similar to game_title
        # using the difflib library
        best_match = difflib.get_close_matches(game_title, steam['name'], n=1, cutoff=0.5)
        
        # If a close match was found
        if best_match:
            # Get the index of the row with the best match
            best_match_index = steam.index[steam['name'] == best_match[0]].tolist()[0]
            
            # Get the row with the best match
            best_match_row = steam.iloc[best_match_index]
            
            # Append a new row to data_frame with the data from the current row of steam_200k
            # and the data from the best match row of steam
            data_frame = data_frame.append(pd.Series(row.tolist() + best_match_row.tolist(), index=data_frame.columns), ignore_index=True)

    # data_frame = merge_dataframes(steam_200k, steam)
    # data_frame = data_frame.apply(process_categories, axis=1)

    split_categories_into_columns_and_populate_with_1_or_0(data_frame)
    data_frame = process_developer(data_frame, 'developer')
    split_genres_into_columns_and_populate_with_1_or_0(data_frame)
    split_platforms_into_columns_and_populate_with_1_or_0(data_frame)


    avg_playtime = data_frame.groupby('game_title')['hours_played'].mean()
    data_frame['avg_play_time'] = data_frame.apply(lambda x: avg_playtime[x['game_title']], axis=1)
    data_frame['avg_play_time'] = data_frame['avg_play_time'].round()
    # data_frame['avg_play_time'] = data_frame['avg_play_time'].astype(int)
    data_frame = remove_rows_where_avg_playtime_is_zero(data_frame)

    create_rating_based_on_hoursplayed(data_frame)
    data_frame = data_frame.drop(columns=['avg_play_time', 'required_age', 'price', 'publisher', 'developer', 'release_date', 'hours_played', 'behavior', 'name', 'categories', 'genres', 'platforms', 'average_playtime', 'appid', 'english', 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings', 'median_playtime', 'owners'])

    # print(data_frame.nunique())
    # print(data_frame.head(100))
    # plot_utils.plotPerColumnDistribution(data_frame, 10, 5)
    # print_avg_playtime(data_frame)
    # Save the DataFrame to a CSV file
    data_frame.to_csv('./data/cleaned_data2.csv', index=False)

def split_categories_into_columns_and_populate_with_1_or_0(data_frame):
    all_categories = set()

    for row in data_frame['categories']:
        categories = row.split(';')
        all_categories.update(categories)

    # Iterate through the dataframe and create a new column for each category
    for category in all_categories:
        data_frame[f'category_{category}'] = 0

    # Iterate through the dataframe and set the value of each category column to 1 if the row has that category
    for index, row in data_frame.iterrows():
        categories = row['categories'].split(';')
        for category in categories:
            data_frame.loc[index, f'category_{category}'] = 1

def split_genres_into_columns_and_populate_with_1_or_0(data_frame):
    all_categories = set()

    for row in data_frame['genres']:
        categories = row.split(';')
        all_categories.update(categories)

    # Iterate through the dataframe and create a new column for each category
    for category in all_categories:
        data_frame[f'genre_{category}'] = 0

    # Iterate through the dataframe and set the value of each category column to 1 if the row has that category
    for index, row in data_frame.iterrows():
        categories = row['genres'].split(';')
        for category in categories:
            data_frame.loc[index, f'genre_{category}'] = 1


def split_platforms_into_columns_and_populate_with_1_or_0(data_frame):
    all_categories = set()

    for row in data_frame['platforms']:
        categories = row.split(';')
        all_categories.update(categories)

    # Iterate through the dataframe and create a new column for each category
    for category in all_categories:
        data_frame[f'platform_{category}'] = 0

    # Iterate through the dataframe and set the value of each category column to 1 if the row has that category
    for index, row in data_frame.iterrows():
        categories = row['platforms'].split(';')
        for category in categories:
            data_frame.loc[index, f'platform_{category}'] = 1

def process_developer(df, developer_column):
    for index, row in df.iterrows():
        developer = row[developer_column]
        if ';' in developer:
            developer = developer.split(';')[0]
            df.loc[index, developer_column] = developer
    
    return df


def create_rating_based_on_hoursplayed(df):
    df['rating'] = df.apply(lambda x: (x['hours_played'] / x['avg_play_time']) * 10, axis=1)
    df['rating'] = df.apply(lambda x: 0 if x['rating'] < 0 else x['rating'], axis=1)
    df['rating'] = df.apply(lambda x: 10 if x['rating'] > 10 else x['rating'], axis=1)

def remove_rows_where_avg_playtime_is_zero(data_frame):
    data_frame = data_frame[data_frame['avg_play_time'] != 0]
    return data_frame

def add_avg_playtime(data_frame):
    avg_playtime = data_frame.groupby('game_title')['hours_played'].mean()
    data_frame['avg_play_time'] = data_frame.apply(lambda x: avg_playtime[x['game_title']], axis=1)
    # data_frame['avg_play_time'] = data_frame['avg_play_time'].round()
    # data_frame['avg_play_time'] = data_frame['avg_play_time'].astype(int)

def prepare_data():
    steam_200k = data_cleaner.clean_data(data_reader.read_data_into_data_frame('./data/data.csv', 'steam_200k'))
    print_utils.print_row_and_col_count(steam_200k)
    # print(data_frame['game_title'].value_counts())
    steam = data_cleaner.clean_data(data_reader.read_data_into_data_frame('./data/steam.csv', 'steam'))
    print_utils.print_row_and_col_count(steam)

    # merge
    data_frame = steam_200k.merge(steam, left_on='game_title', right_on='name')
    print(steam_200k.nunique())
    print("#####")
    print(data_frame.nunique())
    # Save the DataFrame to a CSV file
    # data_frame.to_csv('./data/cleaned_data.csv', index=False)


def merge_dataframes(steam_200k, steam):
    # Create a function to get the best match
    def get_best_match(game_title):
        # Get the best match in df2
        best_match = process.extractOne(game_title, steam['name'])
        return best_match[1]

    # Set the threshold for the similarity
    threshold = 90  
    # Create an empty list to store the matches

    # Check if the match is above the threshold
    if best_match[1] >= threshold:
        # If it is, add the index of the match to the list
        matches.append((index1, best_match[2]))

    # Use the list of matches to merge the dataframes
    merged_df = pd.merge(steam_200k, steam, left_on='index', right_on='index', suffixes=('_steam_200k', '_steam'), how='inner', indicator=True, list=matches)
    return merged_df

# function that accepts 2 strings, normalizes them and returns the similarity score
def is_similarity_above_90(row, left_column, right_column):
    left_value = row[left_column]
    right_value = row[right_column]
    return fuzz.ratio(left_value, right_value) >= 90


# def augment_data(df, num_samples):
#     augmented_data = []
#     existing_user_game_pairs = set(zip(df['user_id'], df['game_id']))
#     for i in range(num_samples):
#         user_id = random.choice(df['user_id'].unique())
#         game_id = random.choice(df['game_id'].unique())
#         while (user_id, game_id) in existing_user_game_pairs:
#             user_id = random.choice(df['user_id'].unique())
#             game_id = random.choice(df['game_id'].unique())
#         rating = round(random.uniform(0, 10),1) # rating between 0-10 as float
#         existing_user_game_pairs.add((user_id, game_id))
#         augmented_data.append([user_id, game_id, rating])
#     return pd.DataFrame(augmented_data, columns=['user_id', 'game_id', 'rating'])

def augment_data(df):
    augmented_data = []
    existing_user_game_pairs = set(zip(df['user_id'], df['game_id']))
    user_steam_uid = {user_id: steam_uid for user_id, steam_uid in zip(df['user_id'], df['steam_uid'])}
    game_title = {game_id: title for game_id, title in zip(df['game_id'], df['game_title'])}
    progress = 1
    total = len(df['steam_uid'].unique())
    current_largest_user_id = df['user_id'].max()
    for steam_uid in df['steam_uid'].unique():
        
        games = df[df['steam_uid'] == steam_uid][['game_id', 'rating']]
        num_samples = random.randint(6, 11)
        for i in range(num_samples):
            new_user_id = uuid.uuid4()
            current_largest_user_id += 1
            print(f"{progress}/{total}")
            user_id = current_largest_user_id
            for index, game in games.iterrows():
                game_id = game['game_id']
                rating = game['rating']
                while (new_user_id, game_id) in existing_user_game_pairs:
                    new_user_id = uuid.uuid4()
                rating = max(0, min(rating + random.uniform(-1, 1), 10))
                existing_user_game_pairs.add((new_user_id, game_id))
                augmented_data.append([user_id, game_id, rating, new_user_id, game_title[game_id]])
        progress += 1
    return pd.DataFrame(augmented_data, columns=['user_id', 'game_id', 'rating', 'steam_uid', 'game_title'])

def main2():
    # print(fuzz.ratio('Counter-Strike Global Offensive', 'Counter-Strike: Global Offensive'))
    data_frame = pd.read_csv('./data/cdata3.csv', delimiter=',')
    data_frame.dataframeName = 'data'
    # data_frame = data_frame.rename(columns={'user_id' : 'steam_uid'})
    
    # data_frame["game_id"] = pd.factorize(data_frame["game_title"])[0]
    # data_frame["user_id"] = pd.factorize(data_frame["steam_uid"])[0]
 
    # append the augmented data to the original data
    data_frame = data_frame.append(augment_data(data_frame), ignore_index=True)
    data_frame.to_csv('./data/augmented-new.csv', index=False)
    # print(data_frame['game_title'].nunique())

def main3():
    df = pd.read_csv('./data/augmented-new.csv', delimiter=',')
    df.dataframeName = 'data'
    df['rating'] = df['rating'].round()
    df['rating'] = df['rating'].clip(0, 10)
    df.to_csv('./data/augmented-rounded.csv', index=False)

def remove_cols_starting_with_category():
    df = pd.read_csv('./data/augmented-rounded.csv', delimiter=',')
    df.dataframeName = 'data'
    new_df = df[["user_id", "game_id", "rating"]]
    # df = df[[col for col in df.columns if not col.startswith("genre") and not col.startswith("category")]]
    new_df.to_csv('./data/augmented-rounded2.csv', index=False)

def get_unique_movies_and_genre_to_new_csv():
    df = pd.read_csv('./data/cdata3.csv', delimiter=',')
    # df = df[[col for col in df.columns if not col.startswith("genre") and not col.startswith("category")]]
        # drop duplicates based on game_id
    df = df.drop_duplicates(subset="game_id")

    # select specific columns
    df = df[["game_id", "game_title"] + [col for col in df.columns if col.startswith("genre_")]]

    # save dataframe to a new csv file
    df.to_csv("raw_data/steam_200k/item.csv", index=False)

def split_rating_to_test_and_train():
    df = pd.read_csv('./data/augmented-rounded2.csv', delimiter=',')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=44, shuffle=True)
      # resetting indices to avoid indexing errors
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    #   # Save train_df to a CSV file
    pd.DataFrame(train_df).to_csv('train_dataset.csv', index=False)

    #   # Save test_df to a CSV file
    pd.DataFrame(test_df).to_csv('test_dataset.csv', index=False)

def remove_duplicate_pairs():
    train_df = pd.read_csv('./raw_data/steam_200k/train_dataset.csv', delimiter=',')
    test_df = pd.read_csv('./raw_data/steam_200k/test_dataset.csv', delimiter=',')
   

    train_df = pd.read_csv(
        './raw_data/steam_200k/train_dataset.csv', header=None,
        names=['user_id', 'game_id', 'ratings'])

    test_df = pd.read_csv(
        './raw_data/steam_200k/test_dataset.csv', header=None,
        names=['user_id', 'game_id', 'ratings'])
    combined_df = pd.concat([train_df, test_df])
    combined_df = combined_df.drop_duplicates(subset=['user_id', 'game_id'], keep='first')
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=44, shuffle=True)
    # Save train_df to a CSV file
    pd.DataFrame(train_df).to_csv('./raw_data/steam_200kv2/train_dataset.csv', index=False)

    #   # Save test_df to a CSV file
    pd.DataFrame(test_df).to_csv('./raw_data/steam_200kv2/test_dataset.csv', index=False)

def check_for_duplicates():
    df = get_games_data('steam_200k')
    # Check for duplicate game id values
    duplicates = df.duplicated(subset=['game id'])

    # Print out the duplicated rows
    print(df[duplicates])

if __name__ == '__main__':
    check_for_duplicates()
