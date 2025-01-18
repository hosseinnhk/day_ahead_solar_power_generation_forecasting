import numpy as np
import pandas as pd
import warnings
import logging
from sklearn.linear_model import LinearRegression, ridge_regression, BayesianRidge, ARDRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR as SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


warnings.filterwarnings('ignore')

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

    # 'solar_azimuth',
    # 'solar_zenith',
    # 'solar_elevation',
    # 'solar_time',

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

'''uncomment to apply PCA'''
# pca = PCA(n_components=4)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cum_sum = np.cumsum(explained_variance_ratio)
# # dim = np.argmax(cum_sum >= 0.95) + 1
# x = pca.transform(x)

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size, :], x[train_size:, :]
Y_train, Y_test = y[0:train_size], y[train_size:]

# poly_features = PolynomialFeatures(degree=1)
# X_train = poly_features.fit_transform(X_train)
# X_test = poly_features.transform(X_test)

mse_scores = []
mae_scores = []
r2score = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = ARDRegression(compute_score=True, n_iter=30)

for train_index, val_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    mse_scores.append(mean_squared_error(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    r2score.append(r2_score(y_val, y_pred))

# joblib.dump(model, "lightgbm_sf_1.pkl")
# print(f"Saved lightgbm model as lightgbm_sf_1.pkl")

print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores))
print('r2 score:', np.mean(r2score))

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)
# pred_test = model.predict(X_test.reshape(-1, 1))
print('test:')
print('mse score:',  mean_squared_error(Y_test, pred_test))
print('mae score:',  mean_absolute_error(Y_test, pred_test))
print('r2 score:', r2_score(Y_test, pred_test))

# res_test = np.concatenate((X_test, Y_test.reshape(-1, 1)), axis=1)
# res_pred_test = np.concatenate((X_test, pred_test.reshape(-1, 1)), axis=1)
# res_test = scaler.inverse_transform(res_test)
# res_pred_test = scaler.inverse_transform(res_pred_test)
#
# Y_test = res_test[:, -1]
# pred_ts = res_pred_test[:, -1]
# pred_ts[pred_ts < 0] = 0

# plt.figure(figsize=(10, 6))
# plt.plot(Y_test)
# plt.plot(pred_ts)
# plt.show()

# pred_0 = pd.DataFrame()
# pred_0['date'] = date_test
# pred_0['time'] = time_test
# pred_0['farm_pred'] = pred_ts
# pred_0['farm_real'] = Y_test
# pred_0.to_csv('sf_pred_0.csv')

# filename = "sgdr_sf_0.pkl"
# joblib.dump(model, filename)
# print(f"Saved sgdr model as {filename}")

