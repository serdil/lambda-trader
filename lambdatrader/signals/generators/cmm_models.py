import xgboost as xgb


class CMMModel:
    def train(self, X, y_close, y_max, y_min):
        raise NotImplementedError


class SklearnCMMModel(CMMModel):

    def __init__(self, model_factory, num_candles, candle_period):
        self.model_factory = model_factory
        self.num_candles = num_candles
        self.candle_period = candle_period

    def train(self, X, y_close, y_max, y_min):
        model_close = self.model_factory.create_model()
        model_close.fit(X, y_close)

        model_max = self.model_factory.create_model()
        model_max.fit(X, y_max)

        model_min = self.model_factory.create_model()
        model_min.fit(X, y_min)

        return model_close, model_max, model_min


class XGBCMMModel(CMMModel):

    def __init__(self, num_candles, candle_period, booster_params,
                 train_ratio=0.8, num_rounds=10000, early_stopping_rounds=100):
        self.num_candles = num_candles
        self.candle_period = candle_period
        self.booster_params = booster_params

        self.train_ratio = train_ratio
        self.num_rounds = num_rounds
        self.early_stopping_rounds = early_stopping_rounds

    def train(self, X, y_close, y_max, y_min):
        gap = self.num_candles

        n_samples = len(X)

        validation_split_ind = int(self.train_ratio * n_samples)

        X_train = X[:validation_split_ind - gap]
        y_max_train = y_max[:validation_split_ind - gap]
        y_min_train = y_min[:validation_split_ind - gap]
        y_close_train = y_close[:validation_split_ind - gap]

        X_val = X[validation_split_ind:]
        y_max_val = y_max[validation_split_ind:]
        y_min_val = y_min[validation_split_ind:]
        y_close_val = y_close[validation_split_ind:]

        dtrain_max = xgb.DMatrix(X_train, label=y_max_train)
        dval_max = xgb.DMatrix(X_val, label=y_max_val)

        dtrain_min = xgb.DMatrix(X_train, label=y_min_train)
        dval_min = xgb.DMatrix(X_val, label=y_min_val)

        dtrain_close = xgb.DMatrix(X_train, label=y_close_train)
        dval_close = xgb.DMatrix(X_val, label=y_close_val)

        watchlist_close = [(dtrain_close, 'train_close'), (dval_close, 'val_close')]
        watchlist_max = [(dtrain_max, 'train_max'), (dval_max, 'val_max')]
        watchlist_min = [(dtrain_min, 'train_min'), (dval_min, 'val_min')]

        bst_close = xgb.train(params=self.booster_params, dtrain=dtrain_close, num_boost_round=self.num_rounds,
                              evals=watchlist_close, early_stopping_rounds=self.early_stopping_rounds)

        bst_max = xgb.train(params=self.booster_params, dtrain=dtrain_max, num_boost_round=self.num_rounds,
                            evals=watchlist_max, early_stopping_rounds=self.early_stopping_rounds)

        bst_min = xgb.train(params=self.booster_params, dtrain=dtrain_min, num_boost_round=self.num_rounds,
                            evals=watchlist_min, early_stopping_rounds=self.early_stopping_rounds)

        model_close = self.XGBSklearnWrapper(bst_close)
        model_max = self.XGBSklearnWrapper(bst_max)
        model_min = self.XGBSklearnWrapper(bst_min)

        return model_close, model_max, model_min

    class XGBSklearnWrapper:
        def __init__(self, booster):
            self.bst = booster

        def predict(self, X):
            dmatrix = xgb.DMatrix(X)
            return self.bst.predict(dmatrix)
