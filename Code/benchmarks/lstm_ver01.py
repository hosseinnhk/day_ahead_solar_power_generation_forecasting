# from feature_engineering import train_sc as X_train
# from feature_engineering import y_train
# from feature_engineering import y_test
# from feature_engineering import test_sc as X_test
from csv_maker import df_pw_21_22 as df_lstm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Input, Dense
from sklearn.metrics import r2_score, PredictionErrorDisplay, mean_squared_error, mean_absolute_percentage_error
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.model_selection import RepeatedKFold, cross_val_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

# print(df_lstm.head())
# df_lstm = df.drop(['hour', 'day', 'month', ], axis=1)  # 'latitude', 'longitude', 'date_col', 'year',
# df_lstm = df_lstm.dropna()
# # df_lstm = df_lstm.set_index('time', drop=True)
# drop_corr_columns = ['cdir', 'ssrd', 'ssr', 'tsr', 'tsrc',  # 'ssrc',
#                      'tisr', 'fdir', 'slhf', 'sshf', 'ssrdc', 'strdc', 'strd', 'uvb', 'ishf',
#                      'str', 'strc', 'alnip', 'alnid', 'flsr', 'ttr', 'ttrc', 'aluvd', 'aluvp',
#                      'd2m', 'msr', 'lsf', 'fal', 'csf', 'tp', 'tco3', 'u10', 'dayIndex']

# df_lstm_dc = df_lstm.drop(drop_corr_columns, axis=1)

# df_lstm = df_lstm.reset_index(drop=False)
df_lstm = df_lstm.reset_index(drop=False)
print(df_lstm.columns)
df_lstm['power'] = df_lstm.pop('power')
df_lstm_notime = df_lstm.loc[:, df_lstm.columns != 'time '] #.reset_index(drop=True)
# print(df_lstm_notime.head(5))
# df_lstm_notime = df_lstm_notime.drop(['time'], axis=1)
# print(df_lstm_notime.tail(50))
# df_lstm_notime = df_lstm_notime.reset_index(drop=True).astype(float)
scaler = StandardScaler()
scaler = scaler.fit(df_lstm_notime)
df_for_training_scaled = scaler.transform(df_lstm_notime)

n_past = 24
n_future = 1
trainX = []
trainY = []
for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training_scaled.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 0])
# print(np.shape(trainY))
# print(np.shape(trainX))

trainX, trainY = np.array(trainX), np.array(trainY)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))
model.compile(optimizer='adam', loss='mse')
model.summary()
history = model.fit(trainX, trainY, epochs=1, batch_size=16, validation_split=0.1, verbose=1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()




n_past = 24
n_hours_for_prediction = 24
prediction = model.predict(trainX[-n_hours_for_prediction:])  # shape = (n, 1)
prediction_copies = np.repeat(prediction, df_lstm_notime.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, -1]

forecast_dates = []
train_dates = pd.to_datetime(df_lstm['time'])
predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_hours_for_prediction, freq='H').tolist()
# print(predict_period_dates)
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'time': np.array(forecast_dates), 'power': y_pred_future})
df_forecast['time'] = pd.to_datetime(df_forecast['time'])
original = df_lstm[['time', 'power']]
# original['time'] = pd.to_datetime(original['time'])
original = original.loc[original['time'] >= '2021-12-28 00:00:00']
# print(original['time'].shape)
plt.plot(original['time'], original['power'])
# plt.plot(df_forecast['time'], df_forecast['power'])
plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


values = df_lstm.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# print(np.shape(scaled))
# frame as supervised learning
reframed = series_to_supervised(scaled, 24, 1)

for i in range(1, 15):
    # for j in range(1, 2):
    reframed.drop(f'var{i}(t)', inplace=True, axis=1)  # +{j}
    # print(f'var{i}(t+{j})')
# print(reframed.columns)
# print(reframed.shape)

values = reframed.values
train = values[:int(0.6 * len(reframed)), :]
test = values[int(0.6 * len(reframed)):, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


model = Sequential()
model.add(LSTM(200, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=200, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                    shuffle=False)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()

yhat = model.predict(test_X)
# print(yhat.shape)
# print(test_X.shape)
test_X = test_X.reshape((test_X.shape[0], 360))
# print(test_X.shape)
inv_yhat = np.concatenate((yhat, test_X[:, -14:]), axis=1)
# print(inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, -14:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
aa = [x for x in range(200)]
plt.plot(aa, inv_y[:200], marker='.', label="actual")
plt.plot(aa, inv_yhat[:200], 'r', label="prediction")
plt.ylabel('normalized PV power generation', size=15)
plt.xlabel('Time step (hour)', size=15)
plt.legend(fontsize=15)
plt.show()

