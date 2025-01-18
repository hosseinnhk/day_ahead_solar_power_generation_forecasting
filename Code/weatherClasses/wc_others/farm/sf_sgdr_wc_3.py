import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor as sgd

df = pd.read_csv('sf_base_regs_1.csv')
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, -1]
x = scaled_df[:, :-1]

train_size = int(len(x) * 0.80)

X_train, X_test, date_test, time_test = x[0:train_size], x[train_size:], date[train_size:], time[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

params = {
    'penalty': 'elasticnet',
    'learning_rate': 'invscaling',
    'alpha': 0.01,
    'eta0': 0.25,
    'max_iter': 15,
    'power_t': 0.10,
    'tol': 3e-05,
    'random_state': 42,
}

model = sgd(**params)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

mse_scores = []
mae_scores = []
r2score = []

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

res_test = np.concatenate((X_test, Y_test.reshape(-1, 1)), axis=1)
res_pred_test = np.concatenate((X_test, pred_test.reshape(-1, 1)), axis=1)
res_test = scaler.inverse_transform(res_test)
res_pred_test = scaler.inverse_transform(res_pred_test)

Y_test = res_test[:, -1]
pred_ts = res_pred_test[:, -1]
pred_ts[pred_ts < 0] = 0

plt.figure(figsize=(10, 6))
plt.plot(Y_test)
plt.plot(pred_ts)
# plt.show()

pred_0 = pd.DataFrame()
pred_0['date'] = date_test
pred_0['time'] = time_test
pred_0['farm_pred'] = pred_ts
pred_0['farm_real'] = Y_test
pred_0.to_csv('sf_pred_1.csv')

filename = "sgdr_sf_1.pkl"
joblib.dump(model, filename)
print(f"Saved sgdr model as {filename}")

