import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def matchType_classify(df):
    def classify(x):
        if 'flare' in x or 'crash' in x or 'normal' in x:
            return 'custom'
        elif 'solo' in x:
            return 'solo'
        elif 'duo' in x:            
            return 'duo'
        else:
            return 'squad'
    
    new_df = df
    new_df['matchType'] = df['matchType'].apply(classify)
    return new_df

def matchType_encoding(df):
    df_OHE = pd.get_dummies(df, columns=['matchType'])
    return df_OHE

def scaling(df, scaler):
    scaler = scaler
    df_scaled = scaler.fit_transform(df)
    df.loc[:,:] = df_scaled[:,:]
    return df

def average_speed(df):
    df['average_speed'] = (df.rideDistance + df.swimDistance + df.walkDistance)/df.matchDuration
    return df['average_speed']

# def headshotKillsPerc(df):
#     if df.kills == 0:
#         return 0
#     else:
#         return df.headshotKills / df.kills

def total_Distance(df):
    df['totalDistance'] = df.rideDistance + df.swimDistance + df.walkDistance
    return df['totalDistance']