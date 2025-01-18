import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor as sgd

df = pd.read_csv('house_base_regs_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time', 'Unnamed: 0'], axis=1)
y = df.actual.values
x = df.drop(['actual'], axis=1)


train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

params = {
    'penalty': 'elasticnet',
    'learning_rate': 'invscaling',
    'alpha': 0,
    'eta0': 0.42,
    'max_iter': 1568,
    'power_t': 0.38,
    # 'tol': 3.35,
    'fit_intercept': True,
    'random_state': 42,
}

model = sgd(**params)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
r2score = []

for train_index, test_index in kf.split(X_train):
    x_train, x_val = X_train.loc[train_index], X_train.loc[test_index]
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


