import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from bayes_opt import BayesianOptimization

df = pd.read_csv('sf_meta_dt.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
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

    'weathercode',

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
], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]


def objective(eta0, power_t, alpha, max_iter, l1_ratio, epsilon):  # , tol
    params = {
        'loss': 'huber',
        'penalty': 'elasticnet',
        'learning_rate': 'invscaling',  # optimal
        'shuffle': False,
        'epsilon': epsilon,
        'eta0': eta0,
        'power_t': power_t,
        'alpha': alpha,
        'l1_ratio': l1_ratio,
        'max_iter': int(max_iter),
        # 'tol': tol,
    }

    mse_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
        x_train, x_val = X_train[train_index], X_train[test_index]
        y_train, y_val = Y_train[train_index], Y_train[test_index]
        model = SGDRegressor(random_state=42, **params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    average_mse = np.mean(mse_scores)
    return -average_mse


param_ranges = {
    'eta0': (0.0001, 1),
    'power_t': (0, 1),
    'alpha': (0, 0.01),
    'l1_ratio': (0, 1),
    'max_iter': (50, 2000),
    'epsilon': (0, 0.1),
    # 'tol': (0, 0.005),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=20, n_iter=100)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")
