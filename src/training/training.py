import xgboost as xgb


def train_model(X_train, y_train, X_val, y_val, params, task):
    if task == "returns":
        model = xgb.XGBRegressor(**params)
    else:
        model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
    return model
