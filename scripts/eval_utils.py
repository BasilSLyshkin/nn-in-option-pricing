import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import product


class Metrics:
    def __init__(self, y_test, X_test, y_bs, y_boost, y_mlp, y_rnn):
        self.columns = ['Black-Scholes', 'Boosting', 'MLP', 'RNN']
        self.predictions = [y_bs, y_boost, y_mlp, y_rnn]
        self.y_test = y_test
        self.X_test = X_test

    def get_full_metrics(self):
        metrics = pd.DataFrame(columns=self.columns, index=['RMSE', 'MAE'])
        for k, model in enumerate(self.columns):
            y_pred = self.predictions[k]
            rmse, mae = self._evaluate(y_pred)
            metrics.loc['RMSE', model] = rmse
            metrics.loc['MAE', model] = mae

        print(metrics.to_latex())

    def get_maturity_metrics(self):
        maturity_nm = ['2', '14', '30']
        metric_nm = ['RMSE', 'MAE']

        index_list = list(product(metric_nm, maturity_nm))
        multi_index = pd.MultiIndex.from_tuples(index_list, names=['Metric', 'Days to Maturity'])

        metrics = pd.DataFrame(columns=['Black-Scholes', 'Boosting', 'MLP', 'RNN'], index=multi_index)

        for mtrt in maturity_nm:
            for k, model in enumerate(self.columns):
                mask = np.isclose(X_test[:,0], int(mtrt)/252, 1e-6)
                if mtrt == '30':
                    mask = self.X_test[:,0]*252 >= 29

                y_pred = self.predictions[k]
                rmse, mae = self._evaluate(y_pred, mask)
                metrics.loc[('RMSE', mtrt), model] = rmse
                metrics.loc[('MAE', mtrt), model] = mae

        print(metrics.to_latex())

    def get_position_metrics(self):

        metric_nm = ['RMSE', 'MAE']
        position_nm = ['OTM', 'ATM', 'ITM']

        index_list = list(product(metric_nm, position_nm))
        multi_index = pd.MultiIndex.from_tuples(index_list, names=['Metric', 'Position'])

        metrics = pd.DataFrame(columns=['Black-Scholes', 'Boosting', 'MLP', 'RNN'], index=multi_index)

        for pstn in position_nm:
            for k, model in enumerate(self.columns):
                if pstn == 'OTM':
                    mask = self.X_test[:, 2]/self.X_test[:,1] < 0.95
                elif pstn == 'ATM':
                    mask = 0.95 <= self.X_test[:, 2]/self.X_test[:,1]
                    mask = mask & (self.X_test[:, 2]/self.X_test[:,1] < 1.05)
                elif pstn == 'ITM':
                    mask = self.X_test[:, 2]/self.X_test[:,1] >= 1.05

                y_pred = self.predictions[k]
                rmse, mae = self._evaluate(y_pred, mask)
                metrics.loc[('RMSE', pstn), model] = rmse
                metrics.loc[('MAE', pstn), model] = mae
        print(metrics.to_latex())

    def _evaluate(self, y_pred, mask=None):

        if mask is None:
            mask = [True]*len(y_pred)

        rmse = np.sqrt(mean_squared_error(self.y_test[mask], y_pred[mask]))
        mae = mean_absolute_error(self.y_test[mask], y_pred[mask])

        return round(rmse, 2), round(mae, 2)