# Feature Engineering

# 팀 플레이어수
from locale import D_FMT


def team_player(df):
    df['team_player']=df.groupId.map(df.groupId.value_counts())
    return df['team_player']

# 총 플레이어 수
def player(df):
    df['player']=D_FMT.matchId.map(df.matchId.value_counts())
    return df['player']


# 총 이동 거리
def total_distance(df):
    df['total_distance']=df['rideDistance']+df['swimDistance']+df['walkDistance']
    return df['total_distance']