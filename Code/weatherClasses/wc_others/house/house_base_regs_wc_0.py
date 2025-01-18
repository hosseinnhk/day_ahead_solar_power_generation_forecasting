import warnings
import joblib
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

df = pd.read_csv('house_wc_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['weathercode', 'snowfall', 'date', 'time',
              # 'is_day', 'shortwave_radiation_instant', 'direct_radiation_instant',
              # 'diffuse_radiation_instant',
              # 'direct_normal_irradiance_instant', 'terrestrial_radiation',
              # 'terrestrial_radiation_instant', 'uv_index_clear_sky', 'uv_index',
              # 'visibility', 'precipitation_probability', 'showers',
              # 'snow_depth'
              ], axis=1)
# print(df.columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

X_test, date_test, time_test = x[train_size:, :], date[train_size:], time[train_size:]
Y_test = y[train_size:]

base_regressors = [
    ('lightgbm', joblib.load('lightgbm_house_0.pkl')),
    # ('extraTree', joblib.load('extraTree_sf_0.pkl')),
    # ('gradientBoost', joblib.load('gradientBoost_sf_0.pkl')),
]

df_pred = pd.DataFrame()
df_pred['date'] = date_test
df_pred['time'] = time_test
for name, model in base_regressors:
    y_pred = model.predict(X_test)
    y_pred[y_pred < 0] = 0
    df_pred[name] = y_pred
df_pred['pvsim'] = X_test[:, -5]
df_pred['actual'] = Y_test
df_pred.to_csv('house_base_regs_0.csv')

