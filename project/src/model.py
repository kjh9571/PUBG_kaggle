from sklearn.metrics import mean_absolute_error

def training(model, X, y):
    model = model
    model.fit(X, y)
    pred_train = model.predict(X)
    mae_train = mean_absolute_error(y, pred_train)
    return mae_train