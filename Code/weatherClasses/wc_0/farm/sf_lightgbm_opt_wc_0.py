import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# Best Hyperparameters:
# feature_fraction: 0.68
# learning_rate: 0.050
# max_depth: 8
# min_child_samples: 3
# min_child_weight: 20
# n_estimators: 1178
# num_iterations: 716
# num_leaves: 127
# reg_alpha: 8.44
# reg_lambda: 1.50
# Best MSE: 0.00384

# Best Hyperparameters:
# min_child_samples: 0.8759
# min_child_weight: 17.4139
# reg_alpha: 7.21
# reg_lambda: 0.92
# Best MSE: 0.00396

file_path = 'D:/pythonProjects/solarPowerForecasting/mainProject/data_prepration/farm/datasets/sf_wc_0.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)
# df = df.drop([
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
#
#     # 'temperature_2m',
#     'dewpoint_2m',
#
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
# ], axis=1)

# df = df.drop([
#     'pvsim',
#     'Lag_24',
#     # 'Lag_25',
#     'Lag_48',
#     'Lag_72',
#
#     # 'solar_azimuth',
#     # 'solar_zenith',
#     # 'solar_elevation',
#     # 'solar_time',
#
#     'shortwave_radiation',
#     'direct_radiation',
#     # 'diffuse_radiation',
#     'direct_normal_irradiance',
#
#     'weathercode',
#
#     'temperature_2m',
#     # 'dewpoint_2m',
#
#     # 'relativehumidity_2m',
#     'surface_pressure',
#
#     'windspeed_10m',
#     'winddirection_10m',
#
#     'cloudcover',
#     # 'cloudcover_low',
#     # 'cloudcover_mid',
#     # 'cloudcover_high',
#     # 'cloud_radiation',
#     #
#     # 'precipitation',
#     # 'rain',
#     'snowfall',
# ], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

x = x[0:train_size]
y = y[0:train_size]

'''uncomment to apply PCA'''
# pca = PCA(n_components=7)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cumsum = np.cumsum(explained_variance_ratio)
# dim = np.argmax(cumsum >= 0.95) + 1
# print(dim)
# x = pca.transform(x)


def objective(
    learning_rate,
    num_iterations,
    max_depth,
    n_estimators,
    num_leaves,
    feature_fraction,
    min_child_samples,
    # reg_alpha,
    # reg_lambda,
    # min_child_weight,
    min_data_in_leaf,
):
    params = {
        'task': 'train',
        'boosting_type': 'goss',
        'objective': 'mape',
        'metric': 'rmse',
        'force_col_wise': True,
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'min_data_in_leaf': int(min_data_in_leaf),
        'min_child_samples': int(min_child_samples),
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
        'verbose': -1,
        'early_stopping_round': 20,
        'feature_pre_filter': True,
        'max_bin': 1000,
    }

    mse_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_val, label=y_val)
        model = lgb.train(params, train_data, valid_sets=[val_data])
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    average_mse = np.mean(mse_scores)
    return -average_mse


param_ranges = {
    # 'num_leaves': 17,  # important
    # 'min_data_in_leaf': 60,  # important
    # 'max_depth': 6,  # important
    # 'learning_rate': 0.072,  # important
    # 'feature_fraction': 0.65,
    # 'min_child_samples': 30,
    # 'n_estimators': 70,
    # "num_iterations": 700,
    # # 'min_child_weight': 20,
    # 'reg_alpha': 6.43,
    # 'reg_lambda': 1.5,

    'num_leaves': (10, 50),
    'learning_rate': (0.05, 0.1),
    'feature_fraction': (0.50, 0.95),
    'min_child_samples': (20, 60),
    'min_data_in_leaf': (10, 80),
    # 'min_child_weight': (0, 20),
    'n_estimators': (30, 100),
    'max_depth': (5, 16),
    "num_iterations": (600, 900),
    # 'reg_alpha':  (5, 7),
    # 'reg_lambda': (1, 2),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=20, n_iter=100)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")
