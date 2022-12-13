

def clean_data(data_frame):
    _remove_duplicates_inplace(data_frame)
    data_frame = _drop_redundant_column(data_frame)
    return data_frame

# check for null values


def _drop_redundant_column(data_frame):
    # if data_frame name is steam_200k, drop the last column
    if data_frame.dataframeName == 'steam_200k':
        return data_frame.drop(columns=['0'])
    return data_frame

def _remove_duplicates_inplace(data_frame):
    # print how many duplicated we have
    print(
        f'Removing {data_frame.duplicated().sum()} duplicated rows')
    data_frame.drop_duplicates(keep='first', inplace=True)
