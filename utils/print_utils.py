def print_row_and_col_count(data_frame):
    nRow, nCol = data_frame.shape
    print(f'There are {nRow} rows and {nCol} columns')



def print_avg_playtime(df):
    # calculate the average play time for each game title
    avg_play_time = df.groupby('game_title')['hours_played'].mean()

    # remove any duplicate game titles
    avg_play_time = avg_play_time.drop_duplicates()

    # reset the index
    avg_play_time = avg_play_time.reset_index()

    # convert the dataframe to a dictionary
    avg_play_time_dict = avg_play_time.to_dict()

    # create a new list of dictionaries with the desired format
    result = []
    for i, game_title in enumerate(avg_play_time_dict['game_title'].values()):
        result.append({game_title: avg_play_time_dict['hours_played'][i]})

    # print the resulting list of dictionaries
    print(result)