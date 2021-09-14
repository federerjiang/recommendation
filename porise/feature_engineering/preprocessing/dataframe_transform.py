from .scalers import GaussRankScaler 
import pandas as pd 


def scale_df(df, columns=[]):
    scaler = GaussRankScaler()
    for col in columns:
        mean = df[col].mean()
        scaler.fit(df[col][~df[col].isna()].values)
        df[col] = df[col].fillna(mean)
        df[col] = scaler.transform(df[col].values)
    return df 


def ohe_df(df, columns=[]):
    for col in columns:
        df[col].fillna(0, inplace=True)
        ohe = pd.get_dummies(df[col])
        ohe_columns = [col + "_ohe_" + str(i) for i in ohe.columns]
        ohe.columns = ohe_columns
        df = pd.concat([df, ohe], axis=1)
    df.drop(columns=columns, inplace=True)
    return df 


def prepare_df(df, num_cols=[], cat_cols=[]):
    if len(cat_cols) > 0:
        df = ohe_df(df, cat_cols)
    if len(num_cols) > 0:
        df = scale_df(df, num_cols)
    return df 