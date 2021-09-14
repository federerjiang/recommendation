import numpy as np
from sklearn.preprocessing import QuantileTransformer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import StandardScaler

# class DataFrameGRScaler(object):
#     """
#     Implements a scaler for dataframe.
#     """

#     def __init__(self, 
#                 df, 
#                 minmax_columns=[],
#                 standard_columns=[],
#                 gaussrank_columns=[],
#                 ):
#         self.df = df 
#         self.minmax_columns = minmax_columns
#         self.standard_columns = standard_columns
#         self.gaussrank_columns = gaussrank_columns

#         self.minmax_scaler = MinMaxScaler()
#         self.standard_scaler = StandardScaler()
#         self.gaussrank_scaler = GaussRankScaler()

#     def fit_transform(self):
#         if len(self.minmax_columns) > 0:
#             self.df[self.minmax_columns] = self.minmax_scaler.fit_transform(self.df[self.minmax_columns])
#         if len(self.standard_columns) > 0:
#             self.df[self.standard_columns] = self.standard_scaler.fit_transform(self.df[self.standard_columns])
#         if len(self.gaussrank_columns) > 0:
#             self.df[self.gaussrank_columns] = self.gaussrank_scaler.fit_transform(self.df[self.gaussrank_columns])
#         return self.df 


class MinMaxScaler(object):
    """Implements min-max scaling."""

    def __init__(self, max=1, min=0):
        self.min = min 
        self.max = max

    def fit(self, x):
        self.x_min = x.min()
        self.x_max = x.max()

    def transform(self, x):
        result = x.astype(float)
        result = (x - self.x_min) / (self.x_max - self.x_min)
        result = result * (self.max - self.min) + self.min 
        return result

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class StandardScaler(object):
    """Implements standard (mean/std) scaling."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x):
        self.mean = x.mean()
        self.std = x.std()

    def transform(self, x):
        result = x.astype(float)
        result -= self.mean
        result /= self.std
        return result

    def inverse_transform(self, x):
        result = x.astype(float)
        result *= self.std
        result += self.mean
        return result

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class GaussRankScaler(object):
    """
    So-called "Gauss Rank" scaling.
    Forces a transformation, uses bins to perform
        inverse mapping.

    Uses sklearn QuantileTransformer to work.
    """

    def __init__(self):
        self.transformer = QuantileTransformer(output_distribution='normal')

    def fit(self, x):
        x = x.reshape(-1, 1)
        self.transformer.fit(x)

    def transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.transform(x)
        return result.reshape(-1)

    def inverse_transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.inverse_transform(x)
        return result.reshape(-1)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

class NullScaler(object):

    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def fit_transform(self, x):
        return self.transform(x)
