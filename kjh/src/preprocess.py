def rm_MissingValue(df):
    new_df = df.dropna(axis=0).copy()
    return new_df

def feature_selection(df):
    new_df = df.drop(columns=['Id','groupId','matchId','killPlace','killPoints',\
        'matchDuration','maxPlace','numGroups','rankPoints','teamKills',\
            'winPoints']).copy()
    return new_df
    