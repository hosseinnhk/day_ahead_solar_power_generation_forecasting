import pandas as pd
import numpy as np
from keras.models import Sequential
import tensorflow as tf
from keras.models import save_model, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv('solarPower.csv', index_col=0, parse_dates=True)
data = df.power[:15000].values
unseen_data = df.power[15000:15300].values

# print(data[-100:])
# plt.figure()
# plt.plot(data[-100:], label='predictions')
# plt.show()
# plt.plot(y_load_pred, label='loaded predictions')
# plt.plot(df[-n_steps:].power.values, label='truth')
# plt.legend()

uns_mean = unseen_data.mean(axis=0)
uns_std = unseen_data.std(axis=0)
uns_data_sc = (unseen_data-uns_mean)/uns_std

mean = data.mean(axis=0)
std = data.std(axis=0)
data_sc = (data - mean) / std

n_lag = 6
n_steps = 12


def prepare_data(data, n_lag):
    X, y = [], []
    for i in range(n_lag, len(data)):
        X.append(data[i-n_lag:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y


X, y = prepare_data(data_sc, n_lag)
X_uns, y_uns = prepare_data(uns_data_sc, n_lag)

train_size = int(len(data) * 0.80)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]
X_test_copy = X_test[-1].copy()


def create_lstm_model():
    model = Sequential()
    model.add(LSTM(200, input_shape=(n_lag, 1), activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    # model.add(LSTM(units=200, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=100, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(LSTM(units=50, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(LSTM(units=100, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


model = create_lstm_model()
history = model.fit(X_train, y_train, epochs=50, batch_size=400, verbose=2, validation_data=(X_test, y_test))
# save_model(model, "model.h5")


def recursive_predict(model, X, n_steps):
    X_copy = X.copy()
    y_pred = np.zeros((n_steps, 1))
    for i in range(n_steps):
        yhat = model.predict(X_copy.reshape(1, n_lag, 1), verbose=0)
        y_pred[i, 0] = yhat
        X_copy[:-1] = X_copy[1:]
        X_copy[-1] = yhat
    return y_pred


y_pred = recursive_predict(model, X_test_copy, n_steps)
y_pred = y_pred * std + mean

for i in range(len(y_pred)):
    if y_pred[i] < 0:
        y_pred[i] = 0

mae = mean_absolute_error(unseen_data[:n_steps], y_pred)
print('Mean absolute error: %.3f' % mae)

# plt.figure()
# plt.plot(y_pred, label='predictions')
# plt.plot(unseen_data[:n_steps], label='truth')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.legend()
# plt.show()

pred_days = 7
uns_y_pred = np.zeros((n_steps, 1))
uns_y_pred_main = []

for i in range(pred_days):
    uns_y_pred = recursive_predict(model, X_uns[24*i], n_steps)
    for value in uns_y_pred:
        uns_y_pred_main.append(value.item())

uns_y_pred_main = np.array(uns_y_pred_main)
uns_y_pred_main = uns_y_pred_main * uns_std + uns_mean

for i in range(len(uns_y_pred_main)):
    if uns_y_pred_main[i] < 0:
        uns_y_pred_main[i] = 0

y_uns = y_uns * uns_std + uns_mean

plt.figure()
plt.plot(uns_y_pred_main, label='forecast2')
plt.plot(y_uns[:pred_days*n_steps], label='real2')
plt.legend()
plt.show()

# model = load_model('model_13580.h5')
# print(model.summary())
# loaded_model.compile(loss='mae', optimizer=Adam(lr=0.001))
# y_load_pred = recursive_predict(loaded_model, X_test_copy, n_steps)
# y_load_pred = y_load_pred * std + mean
# mae = mean_absolute_error(df[-n_steps:].power.values, y_load_pred)
# print('Mean absolute error: %.3f' % mae)