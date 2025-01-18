import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor as sgd, Lasso, Ridge, LinearRegression, LogisticRegression, BayesianRidge, \
    HuberRegressor, RANSACRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor as exr
from sklearn.linear_model import ARDRegression
from sklearn.ensemble import GradientBoostingRegressor as gbr

sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=18)

df = pd.read_csv('sf_meta_dt.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
df = df[df.Power > 0]
df = df.drop([

    'pvsim',
    'Lag_24',
    'Lag_25',
    'Lag_48',
    'Lag_72',

    'solar_azimuth',
    'solar_zenith',
    'solar_elevation',
    'solar_time',

    'shortwave_radiation',
    'direct_radiation',
    'diffuse_radiation',
    'direct_normal_irradiance',

    # 'weathercode',

    'temperature_2m',
    'dewpoint_2m',

    'relativehumidity_2m',
    'surface_pressure',

    'windspeed_10m',
    'winddirection_10m',

    'cloudcover',
    'cloudcover_low',
    'cloudcover_mid',
    'cloudcover_high',
    'cloud_radiation',

    'precipitation',
    'rain',
    'snowfall',
    'pred_lstm',
], axis=1)

# df = df.loc[df['weathercode']==0]
df = df.drop(['weathercode'], axis=1)
print(df.columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
y_max = np.max(y)
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)
print(train_size)
X_train, X_test, date_test, time_test = x[0:train_size], x[train_size:], date[train_size:], time[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

params_sgd = {
    'loss': 'huber',
    'penalty': 'l1',
    'learning_rate': 'invscaling',
    'random_state': 42,
    'shuffle': False,
    # 'l1_ratio': 1,
    'alpha': 0.0016,
    'eta0': 0.3,
    'max_iter': 200,
    'power_t': 0.48,
    'tol': 0.0006,
}
params_gbr = {
    'learning_rate': 0.00768,
    'max_depth': 15,
    'max_features': 18,
    'min_samples_leaf': 9,
    'min_samples_split': 3,
    'n_estimators': 760,
    'subsample': 0.268,
    # Best 'MSE': 0.00456

    # train:
    # mse score: 0.0049547094999567
    # mae score: 0.041857719699343064
    # r2 score: 0.943517408201596
    # test:
    # mse score: 0.008974421723527408
    # mae score: 0.06750517767813846
    # r2 score: 0.87015279733615

    # 'max_depth': 4,
    # 'max_features': 8,
    # 'min_samples_leaf': 5,
    # 'min_samples_split': 14,
    # 'n_estimators': 791,
    # 'learning_rate': 0.038,
    # 'subsample': 0.68,
    # "verbose": 0,
    # 'random_state': 42,
}
params_exr = {
    'max_depth': 4,
    'max_features': 2,
    'max_leaf_nodes': 5,
    'min_samples_leaf': 5,
    'min_samples_split': 5,
    'min_impurity_decrease': 0.0,
    'n_estimators': 100,
    'n_jobs': -1,
    'random_state': 42,
    'verbose': 0,
}

model = sgd(**params_sgd)
# model = LinearRegression()
# model = exr(**params_exr)
# model = SVR(kernel='poly')
# model = gbr(**params_gbr)
# model = RANSACRegressor()
# model = Ridge(alpha=1.0, fit_intercept=True, copy_X=True, max_iter=None, tol=0.0001,
#               solver='saga', positive=False, random_state=None)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
r2score = []

for train_index, test_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[test_index]
    y_train, y_val = Y_train[train_index], Y_train[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    mse = mean_squared_error(y_val, y_pred, squared=False)
    mae = mean_absolute_error(y_val, y_pred)
    r2sc = r2_score(y_val, y_pred)
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2score.append(r2sc)


def mean_absolute_percentage(y_true, y_pred):
    # y_true, y_pred = list(map(float, y_true)), list(map(float, y_pred))
    # assert len(y_true) == len(y_pred), "Input lists must have the same length"
    ymean = np.mean(y_true)
    ape = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]

    # Return the mean of the absolute percentage errors
    mape = (sum(ape) / len(y_true)) * 100
    mape = mape / ymean

    return mape


pred_train = model.predict(X_train)
print('train:')
print('mse score:', np.mean(mse_scores) / np.max(Y_train))
print('mae score:', np.mean(mae_scores) / np.max(Y_train))
print('r2 score:', np.mean(r2score))

# print('traini:')
# print('nRmse score:', mean_squared_error(Y_train, pred_train, squared=False)/np.max(Y_train))
# print('nmae score:', mean_absolute_error(Y_train, pred_train)/np.max(Y_train))
# print('r2 score:', r2_score(Y_train, pred_train))
# print('mape:', mean_absolute_percentage(Y_train, pred_train))

pred_test = model.predict(X_test)
# pred_test, ystd = model.predict(X_test)  #, return_std=True
x1 = 100
x2 = 400

print('test:')
print('nRmse score:', mean_squared_error(Y_test, pred_test, squared=False) / np.max(Y_test))
print('nmae score:', mean_absolute_error(Y_test, pred_test) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, pred_test))
print('mape:', mean_absolute_percentage(Y_test, pred_test))

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test[x1:x2])
# plt.plot(pred_test[x1:x2])
# plt.fill_between(range(0, x2-x1), pred_test[x1:x2] - ystd[x1:x2], pred_test[x1:x2] + ystd[x1:x2], color="pink", alpha=0.5, label="predict std")
# plt.show()

avg = (X_test[:, -2] + X_test[:, -1]) / 2

print('avg:')
print('nmse score:', mean_squared_error(Y_test, avg, squared=False) / np.max(Y_test))
print('nmae score:', mean_absolute_error(Y_test, avg) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, avg))
print('mape:', mean_absolute_percentage(Y_test, avg))

print('lgbm:')
print('nmse score:', mean_squared_error(Y_test, X_test[:, -2], squared=False) / np.max(Y_test))
print('nmae score:', mean_absolute_error(Y_test, X_test[:, -2]) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, X_test[:, -2]))
print('mape:', mean_absolute_percentage(Y_test, X_test[:, -2]))

print('saed:')
print('nmse score:', mean_squared_error(Y_test, X_test[:, -1], squared=False) / np.max(Y_test))
print('nmae score:', mean_absolute_error(Y_test, X_test[:, -1]) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, X_test[:, -1]))
print('mape:', mean_absolute_percentage(Y_test, X_test[:, -1]))

y_persist = [0] * len(Y_test)

for i in range(len(Y_test) - 24):
    y_persist[i + 24] = Y_test[i]

# print('persistance:')
# print('nmse score:', mean_squared_error(Y_test, y_persist, squared=False) / np.max(Y_test))
# print('nmae score:', mean_absolute_error(Y_test, y_persist) / np.max(Y_test))
# print('r2 score:', r2_score(Y_test, y_persist))
# print('mape:', mean_absolute_percentage(Y_test, y_persist))

res_test = np.concatenate((Y_test.reshape(-1, 1), X_test), axis=1)
res_pred_test = np.concatenate((pred_test.reshape(-1, 1), X_test), axis=1)
res_test = scaler.inverse_transform(res_test)
res_pred_test = scaler.inverse_transform(res_pred_test)

Y_test = res_test[:, 0]
pred_ts = res_pred_test[:, 0]
pred_ts[pred_ts < 0] = 0
pred_ts_ex = np.concatenate(([0, 0, 0, 0], pred_ts), axis=0)
Y_test_ex = np.concatenate(([0, 0, 0, 0], Y_test), axis=0)
# pred_ts[Y_test < 50] = 0

## FIG 11
plt.figure(figsize=(22, 6))
plt.plot(Y_test, color='black')
plt.plot(pred_ts, color='red')
plt.ylabel('Solar power (W)')
plt.legend(['Actual', 'Forecast'])
plt.xlabel('Hour')
plt.xlim(-1, 2570)
# plt.savefig(f'figures/all_days.svg')
# plt.show()


## Fig 12
# for i, j in zip(range(0, len(Y_test_ex) - 24, 24), range(24, len(Y_test_ex), 24)):
#     plt.figure(figsize=(10, 6))
#     plt.plot(Y_test_ex[i:j], color='black')
#     plt.plot(pred_ts_ex[i:j], '--', color='red')
#     sns.despine()
#     plt.xlabel('Hour')
#     plt.ylabel('Solar power (W)')
#     plt.xlim(0, 23)
#     plt.legend(['Actual', 'Forecast'])
#     # plt.savefig(f'figures/hour_{i}.png')
#     plt.show()
#     # plt.close()

## Fig 13
y_test_sum = []
y_pred_sum = []
for i, j in zip(range(0, len(Y_test_ex) - 24, 24), range(24, len(Y_test_ex), 24)):
    y_test_sum.append((np.sum(Y_test_ex[i:j] / 1000)))
    y_pred_sum.append((np.sum(pred_ts_ex[i:j] / 1000)))

# bar_width = 0.4
# bar_positions_1 = [i for i in range(len(y_test_sum))]
# bar_positions_2 = [i + bar_width for i in bar_positions_1]

plt.figure(figsize=(22, 6))
# plt.bar(bar_positions_1, y_test_sum, width=bar_width, color='black', alpha=0.9)
# plt.bar(bar_positions_2, y_pred_sum, width=bar_width, color='darkred', alpha=0.8)
plt.plot(y_test_sum, color='black', marker='o')
plt.plot(y_pred_sum, color='maroon', marker='s')
sns.despine()
plt.xlabel('Day')
plt.ylabel('Total daily harvested solar energy (kWh)')
plt.legend(['Actual', 'Forecast'])
plt.xlim(-0.5, 107.1)
# plt.savefig(f'figures/sum_all_days3.svg')
plt.show()

# pred_0 = pd.DataFrame()
# pred_0['date'] = date_test
# pred_0['time'] = time_test
# pred_0['farm_pred'] = pred_ts
# pred_0['farm_real'] = Y_test
# pred_0.to_csv('sf_pred_meta.csv')

# filename = "sgdr_sf_0.pkl"
# joblib.dump(model, filename)
# print(f"Saved sgdr model as {filename}")
