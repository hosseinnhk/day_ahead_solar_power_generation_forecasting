import warnings
import joblib
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

df = pd.read_csv('D:/pythonProjects/solarPowerForecasting/mainProject/data_prepration/farm/datasets/sf_wc_0.csv',
                 parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['weathercode', 'snowfall', 'date', 'time'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.60)

X_train, X_test, date_test, time_test = x[0:train_size, :], x[train_size:, :], date[train_size:], time[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

lgbm_params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'mape',
    'metric': 'rmse',
    'force_col_wise': True,
    'num_leaves': 120,  # important
    'min_data_in_leaf': 20,  # important
    'max_depth': 7,  # important
    'learning_rate': 0.052,
    'feature_fraction': 0.65,
    'min_child_samples': 30,
    'n_estimators': 1000,
    "num_iterations": 700,
    'min_child_weight': 20,
    'reg_alpha': 8.44,
    'reg_lambda': 1.5,
    'n_jobs': -1,
    'class_weight': None,
    'min_split_gain': 0.0,
    'random_state': 42,
    'subsample_freq': 0,
    'verbose': -1,
    'early_stopping_round': 10,
    'feature_pre_filter': True,
    'max_bin': 1000,
}
gbr_params = {
    'max_depth': 4,
    'max_features': 8,
    'min_samples_leaf': 5,
    'min_samples_split': 14,
    'n_estimators': 791,
    'learning_rate': 0.038,
    'subsample': 0.68,
    "verbose": 0,
    'random_state': 42,
    # "loss": "squared_error",
    # "criterion": "friedman_mse",
    # "min_weight_fraction_leaf": 0.0,
    # "min_impurity_decrease": 0.0,
    # "init": None,
    # "alpha": 0.1,
    # "max_leaf_nodes": None,
    # "warm_start": True,
    # "validation_fraction": 0.1,
    # "n_iter_no_change": None,
    # "tol": 0.0001,
    # "ccp_alpha": 0.0
}
ext_params = {
    # 'min_weight_fraction_leaf': 0.1,
    # 'bootstrap': True,
    # 'oob_score': True,
    # 'max_samples': 0.4,
    'max_depth': 37,
    'max_features': 16,
    'max_leaf_nodes': 14,
    'min_samples_leaf': 34,
    'min_samples_split': 10,
    'min_impurity_decrease': 0.0,
    'n_estimators': 2077,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0,
}

base_regressors = [
    ('lightgbm', lgb.LGBMRegressor()),
    ('extraTree', ExtraTreesRegressor(**ext_params)),
    ('gradientBoost', GradientBoostingRegressor(**gbr_params)),
]

df_train = pd.DataFrame()
df_test = pd.DataFrame()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

for name, regressor in base_regressors:
    if name == 'light_GBM':
        for train_index, test_index in kf.split(X_train):
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index]
            train_data = lgb.Dataset(x_train, label=y_train)
            model = lgb.train(lgbm_params, train_data)
        yPred_train = model.predict(X_train)
        yPred_test = model.predict(X_test)
        df_train[name] = yPred_train
        df_test[name] = yPred_test
        filename = f"{name}_sf_0.pkl"
        joblib.dump(regressor, filename)
        print(f"Saved {name} model as {filename}")
    else:
        for train_index, test_index in kf.split(X_train):
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index]
            regressor.fit(x_train, y_train)
        yPred_train = regressor.predict(X_train)
        yPred_test = regressor.predict(X_test)
        df_train[name] = yPred_train
        df_test[name] = yPred_test
        filename = f"{name}_sf_0.pkl"
        joblib.dump(regressor, filename)
        print(f"Saved {name} model as {filename}")

# df_b = pd.DataFrame(pd.concat([df_train, df_test], axis=0))
# df_b['actual'] = y

df_b = df_test
df_b['date'] = date_test
df_b['time'] = time_test
df_b['actual'] = Y_test
# df_b.to_csv('base_regs_0.csv') # test 11/01/23
print('Saved base_regs_0')

for name in ['lightgbm', 'extraTree', 'gradientBoost']:
    print(f'{name} MSE:', mean_squared_error(df_test[name], Y_test))
    print(f'{name} r2:', r2_score(df_test[name], Y_test))
