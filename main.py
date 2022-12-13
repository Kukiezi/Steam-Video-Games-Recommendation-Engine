import os
import utils.plot_utils as plot_utils
import data_cleaner
import data_reader
import utils.print_utils as print_utils
import pandas as pd

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
    data_frame = steam_200k.merge(steam, left_on='game_title', right_on='name')
    data_frame = data_frame.drop(columns=['behavior', 'name', 'average_playtime', 'appid', 'english', 'steamspy_tags', 'achievements', 'positive_ratings', 'negative_ratings', 'median_playtime', 'owners'])

    avg_playtime = data_frame.groupby('game_title')['hours_played'].mean()
    data_frame['avg_play_time'] = data_frame.apply(lambda x: avg_playtime[x['game_title']], axis=1)
    data_frame['avg_play_time'] = data_frame['avg_play_time'].round()
    # data_frame['avg_play_time'] = data_frame['avg_play_time'].astype(int)
    data_frame = remove_rows_where_avg_playtime_is_zero(data_frame)
    create_rating_based_on_hoursplayed(data_frame)
    # print(data_frame.nunique())
    print(data_frame.head(100))
    plot_utils.plotPerColumnDistribution(data_frame, 10, 5)
    # print_avg_playtime(data_frame)
    # Save the DataFrame to a CSV file
    # data_frame.to_csv('./data/cleaned_data.csv', index=False)

    

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

if __name__ == '__main__':
    main()
