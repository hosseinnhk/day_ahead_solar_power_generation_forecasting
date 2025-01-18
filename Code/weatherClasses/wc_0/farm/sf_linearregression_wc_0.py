import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('lightgbm_pred_1061.csv')
df = df.reset_index(drop=True)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[1061:, -1]
x = scaled_df[1061:, -2]

train_size = int(len(x) * 0.4)

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

# poly_features = PolynomialFeatures(degree=2)
# X_train = poly_features.fit_transform(X_train.reshape(-1, 1))
# X_test = poly_features.transform(X_test.reshape(-1, 1))

kf = KFold(n_splits=10, shuffle=True, random_state=42)


model = LinearRegression()

mse_scores = []
mae_scores = []
r2score = []

for train_index, test_index in kf.split(X_train):
    x_train, x_val = X_train[train_index].reshape(-1, 1), X_train[test_index].reshape(-1, 1)
    y_train, y_val = Y_train[train_index], Y_train[test_index]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    mse_scores.append(mean_squared_error(y_val, y_pred))
    mae_scores.append(mean_absolute_error(y_val, y_pred))
    r2score.append(r2_score(y_val, y_pred))

print('train:')
print('mse score:', np.mean(mse_scores))
print('mae score:', np.mean(mae_scores))
print('r2 score:', np.mean(r2score))

pred_test = model.predict(X_test.reshape(-1, 1))
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

