# Feature Engineering
#%%
import pandas as pd
import numpy as np

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

# 컬럼 place화(등수 나열)
def columns_place(list, X, df):
    for i in list:
        X[i + 'Place'] = df.groupby('matchId')[i].rank(method='max', ascending = False)
    
    return X

# matchType 분류
def matchType_classify(df):
    def classify(x):
        if 'flare' in x or 'crash' in x or 'normal' in x:
            return 'event'
        elif 'solo' in x:
            return 'solo'
        elif 'duo' in x:            
            return 'duo'
        else:
            return 'squad'
    
    new_df = df
    new_df['matchType'] = df['matchType'].apply(classify)
    return new_df

# matchType encoding
def matchType_encoding(df):
    df_OHE = pd.get_dummies(df, columns=['matchType'])
    return df_OHE


# 그룹아이디에 따라 그룹화 및 컬럼 평균값 대입.
def columns_grouped_mean(list, X, df):   
    for i in list :
        X['group'+i] = df.groupby('groupId')[i].transform('mean')
        
    return X

def average_weaponsAcquired(df):
    df['average_weaponsAcquired'] = df.weaponsAcquired / (df.matchDuration / 60)
    return df['average_weaponsAcquired']

def average_damage(df):
    df['average_damage'] = df.damageDealt / (df.matchDuration / 60)
    return df['average_damage']




# 지원

# 힐+부스트 당 킬 관여
def healboost_per_kill(df):
    df['healboost_per_kill'] =(df['heals']+df['boosts'])/df['assists']+df['kills']
    return df['healboost_per_kill']

#게임당 거리
def dist_per_game(df):
    df['dist_per_game'] = df['totalDistance']/df['matchDuration']
    return ['dist_per_game']
    
#데미지 비율
def damage_ratio(df):
    df['damage_ratio'] = df['damageDealt']/df['assists']+df['kills']
    return df['damage_ratio']

# 평균등수
def ave_place(df):
    df['ave_maxplace'] = df['killPlace'] / df['maxPlace']
    df['ave_maxplace'].fillna(0, inplace=True)
    df['ave_maxplace'].replace(np.inf, 0, inplace=True)
    return df

#킬당 걸음
def walk_kills(df):
    df['walk_kills'] = df['walkDistance'] / df['kills']
    df['walk_kills'].fillna(0, inplace=True)
    df['walk_kills'].replace(np.inf, 0, inplace=True)
    return df
    
# 총 도움관여 = 어시 + 아군부활 횟수
def support(df):
    df['support'] = df['assists'] + df['revives']
    return df

# 그룹 평균킬
def squad_avg_kill(df):
    df['avg_kill'] = df['squadKills']/df['teamplayer']
    return df


# 솔플 평균킬
def solo_avg_kill(df):
    df['solo_avg_kill'] = df['killPlace']/df['player']
    return df



