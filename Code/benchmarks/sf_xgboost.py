import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


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

def mean_absolute_percentage(y_true, y_pred):
    # y_true, y_pred = list(map(float, y_true)), list(map(float, y_pred))
    # assert len(y_true) == len(y_pred), "Input lists must have the same length"
    ymean = np.mean(y_true)
    ape = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]

    # Return the mean of the absolute percentage errors
    mape = (sum(ape) / len(y_true)) * 100
    mape = mape / ymean

    return mape

print('persistance:')
print('nmse score:', mean_squared_error(df.Power, df.Lag_24, squared=False) / np.max(df.Power))
print('nmae score:', mean_absolute_error(df.Power, df.Lag_24) / np.max(df.Power))
print('r2 score:', r2_score(df.Power, df.Lag_24))
print('mape:', mean_absolute_percentage(df.Power.values, df.Lag_24.values))

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)
X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

xgb_params = {
    # 'colsample_bylevel': 0.28461009351511374,
    # 'colsample_bytree': 0.3370181627021462,
    # 'eta': 0.011681856376962587,
    # 'gamma': 4.769321517668325,
    # 'max_depth': 12,
    # 'min_child_weight': 76.39341189594722,
    # 'reg_alpha': 0.6808286969454159,
    # 'reg_lambda': 70.85479859248183,
    # 'subsample': 0.6964687421743044,

    'colsample_bylevel': 0.2,
    'colsample_bytree': 0.9,
    'eta': 0.007,
    'gamma': 0.0001,
    'max_depth': 6,
    'min_child_weight': 94.81592271521774,
    'reg_alpha': 1.63114242229402,
    'reg_lambda': 29.7088088793984,
    'subsample': 0.9,
}

mse_scores = []
r2_scores = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, val_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dval = xgb.DMatrix(x_val, label=y_val)
    reg = xgb.train(xgb_params, dtrain, 2000)
    # reg = xgb.XGBRegressor(
    #     objective='reg:squarederror',
    #     predictor='cpu_predictor',
    #     n_jobs=-1,
    #     random_state=42,
    #     **xgb_params
    # )
    # reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', verbose=False)
    y_pred = reg.predict(dval)
    mse_scores.append(mean_squared_error(y_val, y_pred))
    r2_scores.append(r2_score(y_val, y_pred))


print('\n_____________________________________')
print('train:')
print('mse score:', np.mean(mse_scores))
print('r2 score:', np.mean(r2_scores), '\n_____________________________________')

dtest = xgb.DMatrix(X_test, label=Y_test)
pred_test = reg.predict(dtest)

# print('test:')
# print('nmse score:', mean_squared_error(Y_test, pred_test) / np.max(Y_test))
# print('r2 score:', r2_score(Y_test, pred_test))

print('test:')
print('nRmse score:', mean_squared_error(Y_test, pred_test, squared=False) / np.max(Y_test))
print('nmae score:', mean_absolute_error(Y_test, pred_test) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, pred_test))
print('mape:', mean_absolute_percentage(Y_test, pred_test))

#
# # display = PredictionErrorDisplay.from_predictions(
# #     y_true=y_test,
# #     y_pred=model.predict(X_test),
# #     kind="actual_vs_predicted",
# #     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
# #     line_kwargs={"color": "tab:red"},
# # )
# # plt.show()
