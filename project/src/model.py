from sklearn.metrics import mean_absolute_error

def training(model, X, y):
    reg = model
    reg.fit(X, y)
    pred_train = reg.predict(X)
    mae_train = mean_absolute_error(y, pred_train)
    return [mae_train, reg]
    