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

df = pd.read_csv('dataset_01.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['weathercode', 'snowfall', 'date', 'time'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

base_regressors = [
    ('lightgbm', joblib.load('lightgbm_sf_1.pkl')),
    # ('extraTree', joblib.load('extraTree_sf_0.pkl')),
    # ('gradientBoost', joblib.load('gradientBoost_sf_0.pkl')),
]

df_pred = pd.DataFrame()
df_pred['date'] = date
df_pred['time'] = time
for name, model in base_regressors:
    y_pred = model.predict(x)
    y_pred[y_pred < 0] = 0
    df_pred[name] = y_pred
# df_pred['pvsim'] = X[:, -5]
df_pred['actual'] = y
df_pred.to_csv('sf_base_regs_1.csv')

for name in ['lightgbm', 'extraTree', 'gradientBoost']:
    print(f'{name} MSE:', mean_squared_error(df_pred[name], y))
    print(f'{name} r2:', r2_score(df_pred[name], y))

train_size = int(len(x) * 0.80)

X_train, X_test, date_test, time_test = x[0:train_size, :], x[train_size:, :], date[train_size:], time[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]


df_train = pd.DataFrame()
df_test = pd.DataFrame()

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = lgb.LGBMRegressor()

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

for train_index, val_index in kf.split(X_train):
    x_train, x_test = X_train[train_index], X_train[val_index]
    y_train, y_test = Y_train[train_index], Y_train[val_index]
    train_data = lgb.Dataset(x_train, label=y_train)
    model = lgb.train(params, train_data)
yPred_train = model.predict(X_train)
yPred_test = model.predict(X_test)
df_train[name] = yPred_train
df_test[name] = yPred_test


# df_b = pd.DataFrame(pd.concat([df_train, df_test], axis=0))
# df_b['actual'] = y

df_b = df_test
df_b['date'] = date_test
df_b['time'] = time_test
df_b['actual'] = Y_test
df_b.to_csv('base_regs_1.csv')
print('Saved base_regs_1')

print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores))
print('r2 score:', np.mean(r2score))

pred_test = model.predict(X_test)
print('test:')
print('mse score:',  mean_squared_error(Y_test, pred_test))
print('mae score:',  mean_absolute_error(Y_test, pred_test))
print('r2 score:', r2_score(Y_test, pred_test))
