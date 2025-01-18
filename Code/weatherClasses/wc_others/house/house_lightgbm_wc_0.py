import numpy as np
import pandas as pd
import joblib
import warnings
import logging
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

df = pd.read_csv('house_wc_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)
df = df.drop(['weathercode', 'snowfall',
#               'is_day', 'shortwave_radiation_instant', 'direct_radiation_instant',
#               'diffuse_radiation_instant',
#               'direct_normal_irradiance_instant', 'terrestrial_radiation',
#               'terrestrial_radiation_instant', 'uv_index_clear_sky', 'uv_index',
#               'visibility', 'precipitation_probability', 'showers',
#               'snow_depth'
              ], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

'''uncomment to apply PCA'''
# pca = PCA(n_components=14)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cumsum = np.cumsum(explained_variance_ratio)
# dim = np.argmax(cumsum >= 0.90) + 1
# x = pca.transform(x)

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size, :], x[train_size:, :]
Y_train, Y_test = y[0:train_size], y[train_size:]

params = {
    'task': 'train',
    'boosting_type': 'goss',
    'objective': 'mape',
    'metric': 'rmse',
    'force_col_wise': True,
    'num_leaves': 110,  # important
    'min_data_in_leaf': 50,  # important
    'max_depth': 6,  # important
    'learning_rate': 0.08,
    'feature_fraction': 1,
    'min_child_samples': 2,
    'n_estimators': 105,
    "num_iterations": 800,
    # 'min_child_weight': 9,
    'reg_alpha': 0.15,
    'reg_lambda': 7,
    'n_jobs': -1,
    'class_weight': None,
    'min_split_gain': 0.0,
    'random_state': 42,
    'subsample_freq': 0,
    'verbose': -1,
    'early_stopping_round': 10,
    'feature_pre_filter': True,
    'max_bin': 1000,
}

mse_scores = []
mae_scores = []
r2score = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = lgb.LGBMRegressor()

for train_index, test_index in kf.split(X_train):
    x_train, x_val = X_train[train_index], X_train[test_index]
    y_train, y_val = Y_train[train_index], Y_train[test_index]
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

joblib.dump(model, "lightgbm_house_0.pkl")
print(f"Saved lightgbm model as lightgbm_house_0.pkl")

print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores))
print('r2 score:', np.mean(r2score))

pred_test = model.predict(X_test)
print('test:')
print('mse score:',  mean_squared_error(Y_test, pred_test))
print('mae score:',  mean_absolute_error(Y_test, pred_test))
print('r2 score:', r2_score(Y_test, pred_test))
