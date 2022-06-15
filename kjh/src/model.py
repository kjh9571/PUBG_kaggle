from sklearn.metrics import mean_absolute_error


def train_test(model, X, y):
    model = model
    model.fit(X, y)
    pred_train = model.predict(X)
    mae_train = mean_absolute_error(y, pred_train)
    