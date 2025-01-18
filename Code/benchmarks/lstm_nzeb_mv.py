import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, PredictionErrorDisplay, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import datetime
import os

sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=20)

# if not os.path.exists("LSTM_1step_Mv"):
#     os.mkdir("LSTM_1step_Mv")

file_path = \
    'D:/TalTechUniversity/solarIradiance/' \
    'forecasting/nZEB_mektory/solar_farm/data_prepration/house/datasets/nZEB_dataset.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop([
    'time',
    'date',
    # 'pvsim',
    # 'Lag_24',
    # 'Lag_25',
    # 'Lag_48',
    # 'Lag_72',

    # 'solar_azimuth',
    # 'solar_zenith',
    # 'solar_elevation',
    # 'solar_time',

    # 'shortwave_radiation',
    # 'direct_radiation',
    # 'diffuse_radiation',
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
    # 'cloud_radiation',

    'precipitation',
    'rain',
    'snowfall',
        ], axis=1)
power = df.pop('Power')
df['Power'] = power

train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size, :]
test_data = df.iloc[train_size:, :]

scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

window_size = 48
# horizon = 24


def prepare_data(data, window_size):
    X, y = [], []
    for i in range(window_size, data.shape[0]):
        X.append(data.iloc[i - window_size:i + 1, :-1].to_numpy())
        y.append(data.iloc[i, -1])
    return np.array(X), np.array(y)


X_train, y_train = prepare_data(train_scaled, window_size)
X_test, y_test = prepare_data(test_scaled, window_size)

# X_test_copy = X_test[-1, :, :].copy()
# print(X_train.shape)
# print(X_test.shape)
# print(X_test_copy.shape)

num_features = X_train.shape[2]


def create_lstm_model():
    model = Sequential()
    model.add(LSTM(30, input_shape=(window_size + 1, num_features), return_sequences=False))
    # model.add(Dropout(0.05))
    model.add(Flatten())
    # model.add(LSTM(units=200))
    # model.add(Dropout(0.25))
    # model.add(Dense(100, activation='tanh'))
    model.add(Dense(1))
    opt = keras.optimizers.Adamax(learning_rate=1e-3, decay=1e-6)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=[tf.keras.metrics.MeanSquaredError()])
    return model


model = create_lstm_model()
callback = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=3)
history = model.fit(X_train, y_train, epochs=50, batch_size=16,  verbose=2, validation_data=(X_test, y_test),
                    callbacks=[callback])
# save_model(model, "model.h5")

pred_train = model.predict(X_train)
pred_test = model.predict(X_test)

pred_test = np.concatenate((X_test[:, -1, :], pred_test), axis=1)
test_scaled = np.concatenate((X_test[:, -1, :], y_test.reshape(-1, 1)), axis=1)

y_pred = scaler.inverse_transform(pred_test)[:, -1]
y_test = scaler.inverse_transform(test_scaled)[:, -1]
y_pred[y_pred < 0] = 0
# y_pred[y_test == 0] = 0
pred_window = 500
print('Test MAE:', mean_absolute_error(y_pred, y_test)/np.max(y_test))
print("Test r2: ", r2_score(y_pred, y_test))

# now = datetime.datetime.now()
# date_string = now.strftime("%Y-%m-%d %H-%M-%S")
#
plt.figure(figsize=(10, 4))
plt.plot(y_pred, '--r', label='prediction')
plt.plot(y_test, 'k', label='truth')
plt.legend(fontsize=14)
sns.despine()
# filename = f"LSTM_1step_Mv/timePlot_{date_string}.svg"
# plt.savefig(filename)
# plt.show()

plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
# plt.title("lossHistory")
# filename = f"LSTM_1step_Mv/lossHistory_{date_string}.svg"
# plt.savefig(filename)
plt.legend()
plt.show()

# display = PredictionErrorDisplay.from_predictions(
#     y_true=y_test_inv,
#     y_pred=y_pred_inv,
#     kind="actual_vs_predicted",
#     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
#     line_kwargs={"color": "tab:red"},
# )
# filename = f"LSTM_1step_Mv/errorDisplay_{date_string}.svg"
# plt.savefig(filename)
# plt.show()

# def recursive_predict(model, X, n_steps):
#     X_copy = X.copy()
#     y_pred = np.zeros((n_steps, 1))
#     for i in range(n_steps):
#         yhat = model.predict(X_copy.reshape(1, window_size, num_features), verbose=0)[0, 0]
#         y_pred[i, 0] = yhat   # test_scaled[i, -1] = y_pred_step
#         X_copy[:-1] = X_copy[1:]
#         X_copy[-1] = yhat
#     return y_pred

# mae = mean_absolute_error(unseen_data[:n_steps], y_pred)
# print('Mean absolute error: %.3f' % mae)
#
# # plt.figure()
# # plt.plot(y_pred, label='predictions')
# # plt.plot(unseen_data[:n_steps], label='truth')
# # plt.legend()
# # plt.show()

# pred_days = 7
# uns_y_pred = np.zeros((n_steps, 1))
# uns_y_pred_main = []
#
# for i in range(pred_days):
#     uns_y_pred = recursive_predict(model, X_uns[24*i], n_steps)
#     for value in uns_y_pred:
#         uns_y_pred_main.append(value.item())
#
# uns_y_pred_main = np.array(uns_y_pred_main)
# uns_y_pred_main = uns_y_pred_main * uns_std + uns_mean

# y_uns = y_uns * uns_std + uns_mean
#
# plt.figure()
# plt.plot(uns_y_pred_main, label='forecast2')
# plt.plot(y_uns[:pred_days*n_steps], label='real2')
# plt.legend()
# plt.show()
