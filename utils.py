import pandas as pd

def load(path, columns=['userId', 'movieId', 'rating']):
    return pd.read_csv(path, sep='\t', names=columns)

def save(df, path):
    df.to_csv(path, index=False, header=False, sep='\t')
