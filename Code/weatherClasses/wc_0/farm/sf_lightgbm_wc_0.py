import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

file_path = 'D:/pythonProjects/solarPowerForecasting/mainProject/data_prepration/farm/datasets/sf_wc_0.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
df = df[df.Power > 0]
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
df = df.drop([

    # 'pvsim',
    # 'Lag_24',
    # 'Lag_25',
    # 'Lag_48',
    # 'Lag_72',

    # 'solar_azimuth',
    # 'solar_zenith',
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
columns = df.columns
print(columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)
y = scaled_df[:, 0]
x = scaled_df[:, 1:]

'''uncomment to apply PCA'''
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cumsum = np.cumsum(explained_variance_ratio)
# dim = np.argmax(cumsum >= 0.90) + 1
# x = pca.transform(x)

index = [0, 0.2, .4, 0.6, 0.8, 1]
i = 4
slice_1 = int(len(x)*index[i])
slice_2 = int(len(x)*index[i+1])

X_train = np.concatenate((x[:slice_1, :], x[slice_2:, :]), axis=0)
X_test = x[slice_1:slice_2, :]
Y_train = np.concatenate((y[:slice_1], y[slice_2:]), axis=0)
Y_test = y[slice_1:slice_2]

params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'mape',
    'metric': 'rmse',
    'force_col_wise': True,
    'feature_fraction': 0.84,
    'learning_rate': 0.0957,
    'max_depth': 8,
    'min_child_samples': 26,
    'min_data_in_leaf': 14,
    'n_estimators': 98,
    'num_iterations': 752,
    'num_leaves': 50,
    'reg_alpha': 5.283,
    'reg_lambda': 1.173,

    'n_jobs': -1,
    'class_weight': None,
    'min_split_gain': 0.0,
    'random_state': 42,
    'subsample_freq': 0,
    'verbose': -1,
    'early_stopping_round': 75,
    'feature_pre_filter': True,
    'max_bin': 50,
}

mse_scores = []
mae_scores = []
r2score = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = lgb.LGBMRegressor()
# best_model = None
# best_score = float('inf')

for train_index, val_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    model = lgb.train(params, train_data, valid_sets=[val_data])
    y_pred = model.predict(x_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    r2sc = r2_score(y_val, y_pred)
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2score.append(r2sc)

lgb.plot_importance(model)
# lgb.plot_split_value_histogram(booster=model, feature=2)
plt.show()
print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores) / np.max(Y_train))
print('r2 score:', np.mean(r2score))
# model = best_model
pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

print('test:')
print('nrmse score:', mean_squared_error(Y_test, pred_test, squared=False) / np.max(Y_test))
print('nmae score:', mean_absolute_error(Y_test, pred_test) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, pred_test))


def mean_absolute_percentage(y_true, y_pred):
    # y_true, y_pred = list(map(float, y_true)), list(map(float, y_pred))
    # assert len(y_true) == len(y_pred), "Input lists must have the same length"
    ymean= np.mean(y_true)
    ape = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]

    # Return the mean of the absolute percentage errors
    mape = (sum(ape) / len(y_true)) * 100
    mape = mape/ymean

    return mape


print('mape:', mean_absolute_percentage(Y_test, pred_test))

y_total = np.concatenate((y[:slice_1], pred_test, y[slice_2:]), axis=0)  #, , pred_train
df_rep = np.concatenate((y_total.reshape(-1, 1), x), axis=1)
df_rep = scaler.inverse_transform(df_rep)
y_rep = df_rep[:, 0]
y_rep[y_rep < 0] = 0
df_p = pd.DataFrame()
df_p['date'] = date[slice_1:slice_2]
df_p['time'] = time[slice_1:slice_2]
# df_p['pred'] = 0
# df_p['pred'].iloc[:train_size] = np.squeeze(pred_train)
# df_p['pred'] = np.squeeze(pred_test)
df_p['pred'] = y_rep[slice_1:slice_2]
df_p['actual'] = df.Power[slice_1:slice_2].values
# df_p.to_csv(f'lightgbm_pred_{slice_1}_{slice_2}.csv')
# print('Saved lightgbm')
