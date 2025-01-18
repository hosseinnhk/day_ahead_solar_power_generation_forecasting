import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import os
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.saving.save import save_model, load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import r2_score, PredictionErrorDisplay, mean_squared_error
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

plt.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 14})
plt.rcParams['legend.frameon'] = 'False'

model = load_model('LSTM_rec_Mv_11025.h5')

if not os.path.exists("LSTM_recur_Mv"):
    os.mkdir("LSTM_recur_Mv")

data = pd.read_csv('onlyDayValues.csv')
data = data.set_index('time')
power = data.pop('power')
data['power'] = power

unseen_data = data.iloc[15000:15300, :]
data = data.iloc[:15000, :]

train_size = int(len(data) * 0.8)
train_data = data.iloc[:train_size, :]
test_data = data.iloc[train_size:, :]

scaler = MinMaxScaler()
train_scaled = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
test_scaled = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
uns_data_scaled = pd.DataFrame(scaler.transform(unseen_data), columns=unseen_data.columns)

window_size = 48
horizon = 24
days = 115


def prepare_data(df, window_size):
    X, y = [], []
    for i in range(window_size, df.shape[0]):
        X.append(df.iloc[i - window_size:i, :].to_numpy())
        y.append(df.iloc[i, -1])
    return np.array(X), np.array(y)


X_train, y_train = prepare_data(train_scaled, window_size)
X_test, y_test = prepare_data(test_scaled, window_size)
X_uns, y_uns = prepare_data(uns_data_scaled, window_size)

# print(train_scaled.iloc[47,:])
# print(X_train[0,-1,:])
# print(np.shape(X_test))

num_features = X_train.shape[2]


def recursive_predict(models, df, hours, days):
    pred = np.zeros((hours * days, 1))
    for day in range(days):
        df_copy = df.copy()
        for hour in range(hours):
            pred[(hour + day * 24), 0] = models.predict(df_copy[(hour + day * 24), :, :]
                                                        .reshape(1, df.shape[1], df.shape[2]),
                                                        verbose=0)
            df_copy[(hour + day * 24) + 1, -1, -1] = pred[(hour + day * 24), 0]
    return pred


y_pred_1step = model.predict(X_test)
y_pred = recursive_predict(model, X_test, horizon, days)
y_pred_con = np.concatenate((y_pred, y_pred_1step[horizon * days:]))

y_pred_prime = np.concatenate((X_test[:, -1, :-1], y_pred_con), axis=1)
y_pred_1step_prime = np.concatenate((X_test[:, -1, :-1], y_pred_1step), axis=1)
y_test_prime = np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1)

y_pred_inv = scaler.inverse_transform(y_pred_prime)[:, -1]
y_pred_inv_1step = scaler.inverse_transform(y_pred_1step_prime)[:, -1]
y_test_inv = scaler.inverse_transform(y_test_prime)[:, -1]

y_pred_inv_1step[y_pred_inv_1step < 0] = 0
y_pred_inv[y_pred_inv < 0] = 0

mae = np.mean(np.abs(y_pred_inv - y_test_inv))
rmse_rec = mean_squared_error(y_pred_inv, y_test_inv, squared=False)
r2 = r2_score(y_pred_inv, y_test_inv)
print('Recursive Test RMSE:', rmse_rec)
print('Recursive Test MAE:', mae)
print("Recursive Test r2: ", r2)

mae_1step = np.mean(np.abs(y_pred_inv_1step - y_test_inv))
rmse_1step = mean_squared_error(y_pred_inv_1step, y_test_inv, squared=False)
r2_1step = r2_score(y_pred_inv_1step, y_test_inv)
print('1 step Test RMSE:', rmse_1step)
print('1 step Test MAE:', mae_1step)
print("1 step Test r2: ", r2_1step)

filename = f"LSTM_rec_Mv_{int(mae)}.h5"
save_model(model, filename)

# now = datetime.datetime.now()
# date_string = now.strftime("%Y-%m-%d %H-%M-%S")

# plt.figure()
# plt.plot(y_pred_inv[:horizon * days], label='recursive prediction')
# plt.plot(y_pred_inv_1step[:horizon * days], label='1 step prediction')
# plt.plot(y_test_inv[:horizon * days], label='truth')
# plt.title("LSTM_recur_Mv")
# plt.legend()
# filename = f"LSTM_recur_Mv/timePlot_{date_string}.svg"
# plt.savefig(filename)
# plt.show()

# plt.figure()
# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.title("lossHistory")
# filename = f"LSTM_recur_Mv/lossHistory_{date_string}.svg"
# plt.savefig(filename)
# plt.legend()
# plt.show()

# fig = plt.figure(figsize=(20, 20), dpi=532)
# display = PredictionErrorDisplay.from_predictions(
#     y_true=y_test_inv,
#     y_pred=y_pred_inv,
#     kind="actual_vs_predicted",
#     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
#     line_kwargs={"color": "tab:red"},
# )
# plt.grid(linewidth=.5)
# plt.title('Multi-step recursive LSTM', fontsize=20)
# plt.xlabel('Predicted values', fontsize=18)
# plt.ylabel('Actual values', fontsize=18)
# plt.tight_layout()
# legend = plt.legend(['R2 score = {:.3f}'.format(r2_score(y_pred_inv, y_test_inv))],
# `````````loc='lower right', fontsize=18)  # frameon=False
# for handle in legend.legendHandles:
#     handle.set_visible(False)
# plt.tight_layout()
# legend.get_frame().set_facecolor('None')
# filename = f"LSTM_recur_Mv/errorDisplay_{date_string}.svg"
# plt.savefig(filename)
# plt.show()
