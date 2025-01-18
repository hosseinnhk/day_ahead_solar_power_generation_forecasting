import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization

# from optuna.samplers import TPESampler

# warnings.filterwarnings('ignore')
#
# df = pd.read_csv('lightgbm_pred_1061.csv')
# df = df.reset_index(drop=True)
# date = df.date.tolist()
# time = df.time.tolist()
# df = df.drop(['date', 'time'], axis=1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_df = scaler.fit_transform(df.values)
#
# y = scaled_df[1061:, -1]
# x = scaled_df[1061:, -2]
#
# train_size = int(len(x) * 0.80)
#
# X_train, X_test = x[0:train_size], x[train_size:]
# Y_train, Y_test = y[0:train_size], y[train_size:]

file_path = \
    'D:/TalTechUniversity/solarIradiance/' \
    'forecasting/nZEB_mektory/solar_farm/data_prepration/farm/datasets/sf_dataset.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df[df.Power > 0]
df = df.drop([
    'time',
    'date',
    'pvsim',
    #     # # 'Lag_24',
    #     # 'Lag_25',
    #     # # 'Lag_48',
    #     # # 'Lag_72',
    #     #
    #     # # 'solar_azimuth',
    #     # # 'solar_zenith',
    #     # # 'solar_elevation',
    #     # # 'solar_time',
    #     #
    #     # # 'shortwave_radiation',
    #     # # 'direct_radiation',
    #     # # 'diffuse_radiation',
    #     # # 'direct_normal_irradiance',
    #     #
    #     'weathercode',
    #     #
    #     # # 'temperature_2m',
    #     'dewpoint_2m',
    #     #
    #     'relativehumidity_2m',
    #     # # 'surface_pressure',
    #     #
    #     # # 'windspeed_10m',
    #     # # 'winddirection_10m',
    #     #
    #     # # 'cloudcover',
    #     'cloudcover_low',
    #     'cloudcover_mid',
    #     'cloudcover_high',
    #     # # 'cloud_radiation',
    #     #
    #     'precipitation',
    #     'rain',
    #     'snowfall',
    #
    #     'solar_azimuth', 'solar_zenith', 'solar_elevation', 'solar_time',
    #     'shortwave_radiation', 'direct_radiation', 'diffuse_radiation',
    #     'direct_normal_irradiance', 'temperature_2m', 'surface_pressure',
    #     'windspeed_10m', 'winddirection_10m', 'cloudcover', 'cloud_radiation'
], axis=1)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

x = x[0:train_size]
y = y[0:train_size]


def objective(
        max_depth,
        # n_estimators,
        eta,
        subsample,
        colsample_bytree,
        colsample_bylevel,
        min_child_weight,
        reg_lambda,
        reg_alpha,
        gamma,
):
    params = {
        'max_depth': int(max_depth),
        # 'n_estimators': n_estimators,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'min_child_weight': min_child_weight,
        'reg_lambda': reg_lambda,
        'reg_alpha': reg_alpha,
        'gamma': gamma,
    }
    # 'max_depth': trial.suggest_int('max_depth', 6, 15),  # Extremely prone to over fitting!
    # 'n_estimators': trial.suggest_int('n_estimators', 400, 4000, 400),  # Extremely prone to over fitting!
    # 'eta': trial.suggest_float('eta', 0.007, 0.013),  # Most important parameter.
    # 'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),
    # 'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 0.9, 0.1),
    # 'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 0.9, 0.1),
    # 'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4),
    # 'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4),  # L2 regularization
    # 'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4),  # L1 regularization
    # 'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),

    mse_scores = []

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)
        reg = xgb.train(params, dtrain, 2000)
        # reg = xgb.XGBRegressor(
        #     objective='reg:squarederror',
        #     predictor='cpu_predictor',
        #     n_jobs=-1,
        #     random_state=42,
        #     **params
        # )
        # reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', verbose=False)
        y_pred = reg.predict(dval)
        mse_scores.append(mean_squared_error(y_val, y_pred))
    return -np.mean(mse_scores)


param_ranges = {
    'max_depth': (6, 15),  # Extremely prone to over fitting!
    # 'n_estimators': (200, 3000),  # Extremely prone to over fitting!
    'eta': (0.007, 0.013),  # Most important parameter.
    'subsample': (0.2, 0.9),
    'colsample_bytree': (0.2, 0.9),
    'colsample_bylevel': (0.2, 0.9),
    'min_child_weight': (1e-4, 100),
    'reg_lambda': (1e-4, 100),  # L2 regularization
    'reg_alpha': (1e-4, 100),  # L1 regularization
    'gamma': (1e-4, 100),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=20, n_iter=100)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")

# train_time = 60 * 10
# study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XGBRegressor')
# study.optimize(objective, timeout=train_time)
#
# print('Number of finished trials: ', len(study.trials), '\n_____________________________________')
# print('Best trial:')
# trial = study.best_trial
#
# print('\tmse value: {}'.format(trial.value))
# print('\tParams: ')
# for key, value in trial.params.items():
#     print('\t\t{}: {}'.format(key, value))

# optuna.visualization.matplotlib.plot_optimization_history(study)
# # plt.savefig('optimization_history.svg', format='svg')
# plt.show()
#
# optuna.visualization.matplotlib.plot_parallel_coordinate(study)
# plt.show()
#
# optuna.visualization.matplotlib.plot_param_importances(study)
# plt.show()

# optuna.visualization.matplotlib.plot_intermediate_values(study)
# plt.show()

