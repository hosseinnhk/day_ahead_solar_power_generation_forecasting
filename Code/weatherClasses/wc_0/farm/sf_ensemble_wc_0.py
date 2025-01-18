import warnings
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import lightgbm as lgb
import logging
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor

logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

df = pd.read_csv('sf_wc_0.csv', parse_dates=['date'], index_col=['date', 'time'])
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
    # 'task': 'train',
    # 'boosting_type': 'dart',
    # 'objective': 'regression_l1',
    # 'metric': 'rmse',
    # 'learning_rate': 0.067,
    # 'feature_fraction': 0.75,
    # 'num_leaves': 253,
    # 'min_child_samples': 3,
    # # 'min_child_weight': 0.91,
    # 'reg_alpha': 0.75,
    # 'reg_lambda': 0.65,
    # 'n_estimators': 1813,
    # # 'colsample_bytree': 0.17,
    # 'num_iterations': 70,
    # 'max_depth': 34,
    # 'force_col_wise': True,
    # 'n_jobs': -1,
    # # 'seed': 0,
    # 'class_weight': None,
    # 'importance_type': 'split',
    # 'min_split_gain': 0.0,
    # 'random_state': 42,
    # 'silent': 'warn',
    # # 'subsample': 0.88,
    # 'subsample_for_bin': 200000,
    # 'subsample_freq': 0,
    # 'verbose': -1,
    # 'task': 'train',
    # 'boosting_type': 'goss',
    # 'objective': 'mape',
    # # 'tree_learner': 'serial',
    # 'metric': 'rmse',
    # 'force_col_wise': True,
    # 'num_leaves': 121,
    # 'learning_rate': 0.028,
    # 'feature_fraction': 0.65,
    # 'min_child_samples': 0,
    # 'n_estimators': 1020,
    # 'max_depth': 7,
    # "num_iterations": 804,
    # 'min_child_weight': 25,
    # 'reg_alpha': 9.5,
    # 'reg_lambda': 1.3,
    # 'n_jobs': -1,
    # 'class_weight': None,
    # 'min_split_gain': 0.0,
    # 'random_state': 42,
    # # 'subsample_for_bin': 200000,
    # 'subsample_freq': 0,
    # 'verbose': -1,
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'mape',
    # 'tree_learner': 'serial',
    'metric': 'rmse',
    'force_col_wise': True,
    'num_leaves': 178,  # 121,
    'learning_rate': 0.048,  # 0.028,
    'feature_fraction': 0.54,  # 0.65,
    'min_child_samples': 3,  # 0,
    'n_estimators': 989,  # 1020,
    'max_depth': 8,  # 7,
    "num_iterations": 839,  # 804,
    'min_child_weight': 21.83,  # 25,
    'reg_alpha': 7.83,  # 9.5,
    'reg_lambda': 1.02,  # 1.3,
    'n_jobs': -1,
    'class_weight': None,
    'min_split_gain': 0.0,
    'random_state': 42,
    # 'subsample_for_bin': 200000,
    'subsample_freq': 0,
    'verbose': -1,
}
gbr_params = {
    # "loss": "squared_error",
    "n_estimators": 928,  # 687,  # 36
    "learning_rate": 0.014,  # 0.007,
    "subsample": 0.75,  # 0.87,
    # "criterion": "friedman_mse",
    "min_samples_split": 7,  # 19,
    "min_samples_leaf": 5,  # 18,
    # "min_weight_fraction_leaf": 0.0,
    "max_depth": 40,  # 26,
    # "min_impurity_decrease": 0.0,
    # "init": None,
    "max_features": 0.22,  # 0.41,
    # "alpha": 0.1,
    # "verbose": 0,
    # "max_leaf_nodes": None,
    # "warm_start": True,
    # "validation_fraction": 0.1,
    # "n_iter_no_change": None,
    # "tol": 0.0001,
    # "ccp_alpha": 0.0
}
ext_params = {
    # 'n_estimators': 2816,
    # 'max_features': 0.59,
    # 'max_depth': 37,
    # "min_samples_split": 1,
    # 'min_samples_leaf': 8,
    # 'min_weight_fraction_leaf': 0.1,
    # 'bootstrap': True,
    # 'oob_score': True,
    # 'max_samples': 0.4,
    # 'warm_start': True,
    'max_depth': 48,
    'max_features': 0.62,
    'min_samples_leaf': 2,
    'min_samples_split': 2,
    'n_estimators': 2928,
}
sgd_params = {
    'learning_rate': 'invscaling',
    'penalty': 'elasticnet',

    # 'alpha': 0.0004, #0.0002,
    # 'eta0': 0.0722, # 0.027,
    # 'max_iter': 6533, #4005,
    # 'power_t': 0.389, #0.107,
    # 'tol': 0.009, #0.004,

    # 'alpha': 0.0,
    # 'eta0': 0.516,
    # 'max_iter': 7175,
    # 'power_t': 0.628,
    # 'tol': 1.0,
    # result:
    # train nRMSE: 8.563727201219172e-07
    # train nMAE: 3.1034175316033187e-07
    # train r2: 0.9922988336217131
    # test  nRMSE: 0.05947815369086972
    # test  nMAE: 0.037811045426567165
    # test  r2: 0.9355905306313709

    # 'alpha': 0.0,
    # 'eta0': 0.694,
    # 'max_iter': 7604,
    # 'power_t': 0.222,
    # 'tol': 3,
    # result:
    # train nRMSE: 8.363298744081196e-07
    # train nMAE: 2.8529057400789043e-07
    # train r2: 0.9926299604744923
    # test  nRMSE: 0.05968173324165136
    # test  nMAE: 0.03830740368544433
    # test  r2: 0.9346775278279582

    'alpha': 0.0022,
    'eta0': 0.27,
    'max_iter': 7968,
    'power_t': 0.512,
    'tol': 3.56,
    'random_state': 42,
}

base_regressors = [
    ('lightgbm', lgb.LGBMRegressor()),
    ('extraTree', ExtraTreesRegressor(random_state=42, **ext_params)),
    ('gradientBoost', GradientBoostingRegressor(random_state=42, **gbr_params)),
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
        # filename = f"{name}_sf_0.pkl"
        # joblib.dump(regressor, filename)
        # print(f"Saved {name} model as {filename}")
    else:
        for train_index, test_index in kf.split(X_train):
            x_train, x_test = X_train[train_index], X_train[test_index]
            y_train, y_test = Y_train[train_index], Y_train[test_index]
            regressor.fit(x_train, y_train)
        yPred_train = regressor.predict(X_train)
        yPred_test = regressor.predict(X_test)
        df_train[name] = yPred_train
        df_test[name] = yPred_test
        # filename = f"{name}_sf_0.pkl"
        # joblib.dump(regressor, filename)
        # print(f"Saved {name} model as {filename}")

# df_b = pd.DataFrame(pd.concat([df_train, df_test], axis=0))
# df_b['actual'] = y
# df_b.to_csv('base_regs_0.csv')

for name in df_test.columns:
    print(f'{name} MSE:', mean_squared_error(df_test[name], Y_test))
    print(f'{name} r2:', r2_score(df_test[name], Y_test))

meta_learner = SGDRegressor(**sgd_params)
for train_index, test_index in kf.split(df_train):
    x_train, x_test = df_train.iloc[train_index], df_train.iloc[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    meta_learner.fit(x_train, y_train)

pred_tr = meta_learner.predict(df_train)
pred_ts = meta_learner.predict(df_test)
# filename = "sgdr_sf_0.pkl"
# joblib.dump(meta_learner, filename)
# print(f"Saved sgdr model as {filename}")

res_test = np.concatenate((Y_test.reshape(-1, 1), X_test), axis=1)
res_pred_test = np.concatenate((pred_ts.reshape(-1, 1), X_test), axis=1)
res_test = scaler.inverse_transform(res_test)
res_pred_test = scaler.inverse_transform(res_pred_test)

Y_test = res_test[:, 0]
pred_ts = res_pred_test[:, 0]
# pred_tr[pred_tr < 0] = 0
pred_ts[pred_ts < 0] = 0

farm_cap = 30600
print('final results:')
print('train nRMSE:', mean_squared_error(pred_tr, Y_train, squared=False) / farm_cap)
print("train nMAE:", mean_absolute_error(pred_tr, Y_train) / farm_cap)
print('train r2:', r2_score(pred_tr, Y_train))

print("test  nRMSE:", mean_squared_error(pred_ts, Y_test, squared=False) / farm_cap)
print("test  nMAE:", mean_absolute_error(pred_ts, Y_test) / farm_cap)
print("test  r2:", r2_score(pred_ts, Y_test))

plt.figure(figsize=(10, 6))
plt.plot(Y_test[100:])
plt.plot(pred_ts[100:])
# plt.show()

# pred_0 = pd.DataFrame()
# pred_0['date'] = date_test
# pred_0['time'] = time_test
# pred_0['farm_pred'] = pred_ts
# pred_0['farm_real'] = Y_test
# pred_0.to_csv('sf_pred_0.csv')
