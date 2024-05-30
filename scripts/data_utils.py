import torch
from torch.utils.data import Dataset

from tqdm.notebook import tqdm

import numpy as np

class DataSet(Dataset):
    """
    Technical dataset class for PyTorch DataLoader
    """

    def __init__(self, X_y_tuple):

        x_tmp, y_tmp = X_y_tuple
        self.x = torch.tensor(x_tmp, dtype=torch.float32)
        self.y = torch.tensor(y_tmp, dtype=torch.float32)

        self.n_samples = y_tmp.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def get_timeseries(df):
    """
    Transforms the data into timeseries for RNN
    :param df: Initial DataFrame
    :return: X of shape (N, sequence_length, 5), y of shape (N,)
    """
    features = ['time_to_maturity', 'strike', 'spot_price', 'risk_free_rate', 'volatility']
    target = 'option_payout'
    sequence_length = 14

    X = []
    y = []

    for ticker in tqdm(df['ticker'].unique()):
        ticker_data = df[df['ticker'] == ticker]

        for i in range(sequence_length, len(ticker_data)):
            X.append(ticker_data[features].iloc[i - sequence_length: i].values)

            # Discounting
            discounter = np.exp(-ticker_data['time_to_maturity'].iloc[i - sequence_length : i].values * ticker_data['risk_free_rate'].iloc[i - sequence_length : i].values)
            y.append(ticker_data[target].iloc[i] * discounter)

    X = np.array(X)
    y = np.array(y)

    return X, y