# Feature Engineering

# 팀 플레이어수
def team_player(df):
    df['team_player']=df.groupId.map(df.groupId.value_counts())
    return df['team_player']

# 총 플레이어 수
def player(df):
    df['player']=df.matchId.map(df.matchId.value_counts())
    return df['player']

# 총 이동 거리
def total_distance(df):
    df['total_distance']=df['rideDistance']+df['swimDistance']+df['walkDistance']
    return df['total_distance']
  
def scaling(df, scaler, col_list):
    scaler = scaler
    temp = scaler.fit_transform(df.loc[:, col_list])
    for i in range(len(col_list)):
        df[col_list[i]] = temp[:,i]
    return df

def headshotKillsPerc(row):
    if row['kills'] == 0:
        return 0
    else:
        return row['headshotKills'] / row['kills']

def columns_place(list, X, df):
    for i in list:
        X[i + 'Place'] = df.groupby('matchId')[i].rank(method='max', ascending = False)
    
    return X