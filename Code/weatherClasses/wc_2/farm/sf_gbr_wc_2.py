import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

file_path = \
    'D:/TalTechUniversity/solarIradiance/forecasting/nZEB_mektory/solar_farm/data_prepration/farm/datasets/sf_wc_2.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop([
    'date',
    'time',
    # 'pvsim',
    # 'Lag_24',
    'Lag_25',
    'Lag_48',
    # 'Lag_72',

    'solar_azimuth',
    'solar_zenith',
    'solar_elevation',
    'solar_time',

    # 'shortwave_radiation',
    # 'direct_radiation',
    'diffuse_radiation',
    # 'direct_normal_irradiance',

    'weathercode',

    # 'temperature_2m',
    'dewpoint_2m',

    'relativehumidity_2m',
    # 'surface_pressure',

    # 'windspeed_10m',
    # 'winddirection_10m',

    # 'cloudcover',
    'cloudcover_low',
    'cloudcover_mid',
    'cloudcover_high',
    'cloud_radiation',

    'precipitation',
    'rain',
    'snowfall',
], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size, :], x[train_size:, :]
Y_train, Y_test = y[0:train_size], y[train_size:]

params = {
    'max_depth': 4,
    'max_features': 8,
    'min_samples_leaf': 5,
    'min_samples_split': 14,
    'n_estimators': 791,
    'learning_rate': 0.038,
    'subsample': 0.68,
    "verbose": 0,
    'random_state': 42,
}

mse_scores = []
mae_scores = []
r2score = []

model = gbr(**params)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[test_index]
    y_train, y_val = Y_train[train_index], Y_train[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2sc = r2_score(y_val, y_pred)
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2score.append(r2sc)
print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores))
print('r2 score:', np.mean(r2score))

pred_test = model.predict(X_test)
print('test:')
print('mse score:',  mean_squared_error(Y_test, pred_test))
print('mae score:',  mean_absolute_error(Y_test, pred_test))
print('r2 score:', r2_score(Y_test, pred_test))
