import pandas as pd

def load_data(x):
    base_path = 'data/raw/'
    df = pd.read_csv(base_path + x + '_V2.csv')
    return df