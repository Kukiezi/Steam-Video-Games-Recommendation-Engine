import pandas as pd


def read_data_into_data_frame(file_path, name, rows=None):
    # steam-200k.csv has 199999 rows in reality, but we are only loading/previewing the first 1000 rows
    data_frame = pd.read_csv(file_path, delimiter=',', nrows=rows)
    data_frame.dataframeName = name
    _assign_column_names(data_frame)
    return data_frame


def _assign_column_names(data_frame):
    if data_frame.dataframeName == 'steam_200k':
        data_frame.columns = ['user_id', 'game_title', 'behavior', 'hours_played', '0']
