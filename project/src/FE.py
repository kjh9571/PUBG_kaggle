# Feature Engineering
def scaling(df, scaler, col_list):
    scaler = scaler
    temp = scaler.fit_transform(df.loc[:, col_list])
    for i in range(len(col_list)):
        df[col_list[i]] = temp[:,i]
    return df

