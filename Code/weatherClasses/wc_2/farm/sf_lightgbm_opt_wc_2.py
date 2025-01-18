import numpy as np
import pandas as pd
import warnings
import joblib
import logging
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA

logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

# Best Hyperparameters:
# feature_fraction: 0.52
# learning_rate: 0.054
# max_depth: 11
# min_child_samples: 33
# n_estimators: 1406
# num_iterations: 590
# num_leaves: 147
# reg_alpha: 7.16
# reg_lambda: 0.49
# Best MSE: 0.00833

'''with feature drops'''
# Best Hyperparameters:
# feature_fraction: 0.666
# learning_rate: 0.0976
# max_depth: 5
# min_child_samples: 87
# min_data_in_leaf: 14
# n_estimators: 1198
# num_iterations: 641
# num_leaves: 164
# reg_alpha: 4.989
# reg_lambda: 0.657
# Best MSE: 0.00952

file_path = \
    'D:/TalTechUniversity/solarIradiance/forecasting/nZEB_mektory/solar_farm/data_prepration/farm/datasets/sf_wc_2.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)
df = df.drop([
#     # 'pvsim',
#     # 'Lag_24',
#     'Lag_25',
#     # 'Lag_48',
#     # 'Lag_72',
#
#     'solar_azimuth',
#     'solar_zenith',
#     'solar_elevation',
#     'solar_time',
#
#     # 'shortwave_radiation',
#     # 'direct_radiation',
#     'diffuse_radiation',
#     # 'direct_normal_irradiance',
#
#     'weathercode',

    'temperature_2m',
    # 'dewpoint_2m',

#     'relativehumidity_2m',
#     # 'surface_pressure',
#
#     # 'windspeed_10m',
#     # 'winddirection_10m',
#
#     # 'cloudcover',
#     'cloudcover_low',
#     'cloudcover_mid',
#     'cloudcover_high',
#     'cloud_radiation',
#
#     'precipitation',
#     'rain',
#     'snowfall',
], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size, :], x[train_size:, :]
Y_train, Y_test = y[0:train_size], y[train_size:]

'''uncomment to apply PCA'''
# pca = PCA(n_components=14)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cum_sum = np.cumsum(explained_variance_ratio)
# dim = np.argmax(cum_sum >= 0.95) + 1
# # print(dim)
# x = pca.transform(x)


def objective(
        learning_rate,
        num_iterations,
        min_data_in_leaf,
        max_depth,
        n_estimators,
        num_leaves,
        # feature_fraction,
        min_child_samples,
        # reg_alpha,
        # reg_lambda,
        # min_child_weight,
):
    params_opt = {
        'task': 'train',
        'boosting_type': 'goss',
        'objective': 'mape',
        'metric': 'rmse',
        'force_col_wise': True,
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        # 'feature_fraction': feature_fraction,
        'min_child_samples': int(min_child_samples),
        'min_data_in_leaf': int(min_data_in_leaf),
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        "num_iterations": int(num_iterations),
        # 'min_child_weight': min_child_weight,
        # 'reg_alpha': reg_alpha,
        # 'reg_lambda': reg_lambda,
        'n_jobs': -1,
        'class_weight': None,
        'min_split_gain': 0.0,
        'random_state': 42,
        'subsample_freq': 0,
        'verbose': -100,
        'early_stopping_round': 20,
        'feature_pre_filter': True,
        'max_bin': 1000,
    }

    mse_scores_opt = []
    kf_opt = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index_opt, val_index_opt in kf_opt.split(X_train):
        x_train_opt, x_val_opt = X_train[train_index_opt], X_train[val_index_opt]
        y_train_opt, y_val_opt = Y_train[train_index_opt], Y_train[val_index_opt]
        train_data_opt = lgb.Dataset(x_train_opt, label=y_train_opt)
        val_data_opt = lgb.Dataset(x_val_opt, label=y_val_opt)
        model_opt = lgb.train(params_opt, train_data_opt, valid_sets=[val_data_opt])
        y_pred_opt = model_opt.predict(x_val_opt)
        mse_scores_opt.append(mean_squared_error(y_val_opt, y_pred_opt))
    average_mse_opt = np.mean(mse_scores_opt)
    return -average_mse_opt


param_ranges = {
    'num_leaves': (20, 200),
    'learning_rate': (0.0001, 0.1),
    # 'feature_fraction': (0.2, 1),
    'min_child_samples': (0, 100),
    'min_data_in_leaf': (10, 80),
    # 'min_child_weight': (0, 50),
    'n_estimators': (600, 1500),
    'max_depth': (3, 20),
    "num_iterations": (100, 1500),
    # 'reg_alpha': (0, 10),
    # 'reg_lambda': (0, 1000),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=10, n_iter=100)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

best_param = {}
print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")
    best_param[key] = value
print(f"Best MSE: {best_mse}")

keys_to_convert = ['num_iterations', 'num_leaves', 'min_child_samples', 'n_estimators', 'max_depth', 'num_iterations',
                   'min_data_in_leaf']

for key in keys_to_convert:
    if key in best_param:
        best_param[key] = int(best_param[key])

params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'mape',
    'metric': 'rmse',
    'force_col_wise': True,
    'n_jobs': -1,
    'class_weight': None,
    'min_split_gain': 0.0,
    'random_state': 42,
    'subsample_freq': 0,
    'verbose': -100,
    'early_stopping_round': 20,
    'feature_pre_filter': True,
    'max_bin': 1000,
}
params.update(best_param)
mse_scores = []
mae_scores = []
r2score = []

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = lgb.LGBMRegressor()

for train_index, val_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    model = lgb.train(params, train_data, valid_sets=[val_data])
    y_pred = model.predict(x_val)
    mse_scores.append(mean_squared_error(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    r2score.append(r2_score(y_val, y_pred))

# joblib.dump(model, "trained_models/lightgbm_sf_1.pkl")
print(f"Saved lightgbm model as lightgbm_sf_1.pkl")

print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores))
print('r2 score:', np.mean(r2score))

pred_test = model.predict(X_test)
print('test:')
print('mse score:', mean_squared_error(Y_test, pred_test))
print('mae score:', mean_absolute_error(Y_test, pred_test))
print('r2 score:', r2_score(Y_test, pred_test))

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")
    best_param[key] = value
print(f"Best MSE: {best_mse}")

""" random search """

# import numpy as np
# import pandas as pd
# import warnings
# import joblib
# import logging
# import lightgbm as lgb
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.decomposition import PCA
# import random

# logging.getLogger('lightgbm').setLevel(logging.ERROR)
# warnings.filterwarnings('ignore')
#
# df = pd.read_csv('sf_wc_1.csv', parse_dates=['date'], index_col=['date', 'time'])
# df = df.reset_index(drop=True)
# df = df.drop(['weathercode', 'snowfall'], axis=1)
#
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_df = scaler.fit_transform(df.values)
#
# y = scaled_df[:, 0]
# x = scaled_df[:, 1:]
#
# train_size = int(len(x) * 0.80)
#
# X_train, X_test = x[0:train_size, :], x[train_size:, :]
# Y_train, Y_test = y[0:train_size], y[train_size:]
#
# '''uncomment to apply PCA'''
# # pca = PCA(n_components=14)
# # X_pca = pca.fit_transform(x)
# # explained_variance_ratio = pca.explained_variance_ratio_
# # cum_sum = np.cumsum(explained_variance_ratio)
# # dim = np.argmax(cum_sum >= 0.95) + 1
# # # print(dim)
# # x = pca.transform(x)
#
#
# def objective(params):
#     learning_rate = params['learning_rate']
#     num_iterations = params['num_iterations']
#     max_depth = params['max_depth']
#     n_estimators = params['n_estimators']
#     num_leaves = params['num_leaves']
#     feature_fraction = params['feature_fraction']
#     min_child_samples = params['min_child_samples']
#     reg_alpha = params['reg_alpha']
#     reg_lambda = params['reg_lambda']
#     # min_child_weight = params['min_child_weight']
#
#     params_opt = {
#         'task': 'train',
#         'boosting_type': 'goss',
#         'objective': 'mape',
#         'metric': 'rmse',
#         'force_col_wise': True,
#         'num_leaves': int(num_leaves),
#         'learning_rate': learning_rate,
#         'feature_fraction': feature_fraction,
#         'min_child_samples': int(min_child_samples),
#         'n_estimators': int(n_estimators),
#         'max_depth': int(max_depth),
#         "num_iterations": int(num_iterations),
#         # 'min_child_weight': min_child_weight,
#         'reg_alpha': reg_alpha,
#         'reg_lambda': reg_lambda,
#         'n_jobs': -1,
#         'class_weight': None,
#         'min_split_gain': 0.0,
#         'random_state': 42,
#         'subsample_freq': 0,
#         'verbose': -100,
#         # 'early_stopping_round': 20,
#         'feature_pre_filter': True,
#         'max_bin': 1000,
#     }
#
#     mse_scores_opt = []
#     kf_opt = KFold(n_splits=10, shuffle=True, random_state=42)
#
#     for train_index_opt, val_index_opt in kf_opt.split(X_train):
#         x_train_opt, x_val_opt = X_train[train_index_opt], X_train[val_index_opt]
#         y_train_opt, y_val_opt = Y_train[train_index_opt], Y_train[val_index_opt]
#         train_data_opt = lgb.Dataset(x_train_opt, label=y_train_opt)
#         val_data_opt = lgb.Dataset(x_val_opt, label=y_val_opt)
#         model_opt = lgb.train(params_opt, train_data_opt)  # , valid_sets=[val_data_opt]
#         y_pred_opt = model_opt.predict(x_val_opt)
#         mse_opt = mean_squared_error(y_val_opt, y_pred_opt)
#         mse_scores_opt.append(mse_opt)
#     average_mse_opt = np.mean(mse_scores_opt)
#     return average_mse_opt
#
#
# param_ranges = {
#     'num_leaves': (20, 200),
#     'learning_rate': (0.001, 0.1),
#     'feature_fraction': (0.2, 1),
#     'min_child_samples': (0, 100),
#     # 'min_child_weight': (0, 20),
#     'n_estimators': (600, 1500),
#     'max_depth': (3, 20),
#     "num_iterations": (500, 1500),
#     'reg_alpha': (0, 10),
#     'reg_lambda': (0, 1),
# }
#
# num_random_search_iter = 1000
# best_mse = float('inf')
# best_params = {}
#
# for _ in range(num_random_search_iter):
#     random_params = {
#         param: random.uniform(*param_ranges[param])
#         for param in param_ranges
#     }
#     mse = objective(random_params)
#     if mse < best_mse:
#         best_mse = mse
#         best_params = random_params
#         print("Best MSE so far:", best_mse)
#         print("Best Hyperparameters so far:")
#         print(best_params)
#
# print("Best Hyperparameters:")
# print(best_params)
# print("Best MSE:", best_mse)
