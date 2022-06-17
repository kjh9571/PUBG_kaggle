# Feature Engineering

# 팀 플레이어수

def team_player(df):
    df['team_player']=df.groupId.map(df.groupId.value_counts())
    return df['team_player']

    