import pandas as pd
import numpy as np

df_features = pd.read_csv('./dataset/fscore_features.csv')
df_features.columns = ['feature', 'fscore']

def get_data(limit=None):
    df = pd.read_csv('./dataset/df_results.csv')
    X = df.drop(columns=['gallstoneStatus', 'Unnamed: 0'])
    Y = df['gallstoneStatus']
    # filter by anova features fscore > 0.70
    X = X[df_features['feature']]
    
    return X, Y, df