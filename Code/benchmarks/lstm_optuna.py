import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.saving.save import save_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import L1, L2
from keras.optimizers import RMSprop, SGD
from sklearn.metrics import r2_score, PredictionErrorDisplay
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
# from featureSelection import onlyDayValues as data
import tensorflow as tf
import datetime
import os

# if not os.path.exists("LSTM_recur_Mv"):
#     os.mkdir("LSTM_recur_Mv")

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
days = 6


def prepare_data(df, slide_size):
    X, y = [], []
    for i in range(slide_size, df.shape[0]):
        X.append(df.iloc[i-slide_size:i, :].to_numpy())
        y.append(df.iloc[i, -1])
    return np.array(X), np.array(y)


X_train, y_train = prepare_data(train_scaled, window_size)
X_test, y_test = prepare_data(test_scaled, window_size)
X_uns, y_uns = prepare_data(uns_data_scaled, window_size)
# print(X_train.shape)
# print(X_test.shape)
# print(X_test_copy.shape)

num_features = X_train.shape[2]


def create_lstm_model(trial):
    n_layers = 2  # trial.suggest_int("n_layers", 1, 3)
    models = Sequential()
    lstm_activation_functions = ['tanh', 'sigmoid', 'relu']
    dense_activation_functions = ['relu', 'sigmoid', 'linear']

    for i in range(n_layers):
        units = trial.suggest_int(f"units_layer{i + 1}", 30, 400)
        units2 = trial.suggest_int(f"units_layer{i + 1}", 30, 250)
        dropout_rate = trial.suggest_float(f"dropout_rate_layer{i + 1}", 0.0, 0.5)
        lstm_activation = trial.suggest_categorical(f"lstm_activation_layer{i + 1}", lstm_activation_functions)

        if i == 0:
            models.add(LSTM(units, input_shape=(window_size, num_features), activation=lstm_activation, return_sequences=True))
        else:
            models.add(LSTM(units2, activation=lstm_activation, return_sequences=(i < n_layers - 1)))

        models.add(Dropout(dropout_rate))

    dense_units = trial.suggest_int("dense_units", 50, 200)
    dense_activation = trial.suggest_categorical("dense_activation", dense_activation_functions)
    models.add(Dense(dense_units, activation=dense_activation))
    models.add(Dense(1))

    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    optimizer = RMSprop(learning_rate=learning_rate)
    models.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])

    return models


def objective(trial):
    model = create_lstm_model(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1,
                        validation_data=(X_test, y_test), callbacks=[early_stop])
    val_loss = history.history["val_loss"][-1]
    return val_loss


sampler = TPESampler()
study = optuna.create_study(sampler=sampler, direction="minimize")
study.optimize(objective, n_trials=25, n_jobs=-1)

print("Best hyper parameters:", study.best_params)
print("Best loss:", study.best_value)

# best_model = create_lstm_model(optuna.trial.FixedTrial(study.best_params))
# early_stop = EarlyStopping(monitor='val_loss', patience=10)
# best_model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2,
#                validation_data=(X_test, y_test), callbacks=[early_stop])
#
#
# def recursive_predict(models, df, n_steps, day):
#     pred = np.zeros((n_steps * days, 1))
#     for step in range(day):
#         df_copy = df.copy()
#         for i in range(n_steps):
#             pred[(i+step * 24), 0] = \
#                 models.predict(df_copy[(i+step * 24), :, :].reshape(1, df.shape[1], df.shape[2]), verbose=0)
#             df_copy[(i+step * 24)+1, -1, -1] = pred[(i+step * 24), 0]
#     return pred
#
#
# y_pred_1step = best_model.predict(X_test)
# # y_pred_1step_ls = [item for sublist in y_pred_1step for item in sublist]
# # print(y_pred_1step_ls[:horizon])
# y_pred = recursive_predict(best_model, X_test, horizon, days)
# # y_pred_ls = [item for sublist in y_pred for item in sublist]
# # print(y_pred_ls)
# y_pred_con = np.concatenate((y_pred, y_pred_1step[horizon * days:]))
# # print(y_pred_con[:horizon])
# # print(y_pred_1step_ls[:horizon])
#
# y_pred_prime = np.concatenate((X_test[:, -1, :-1], y_pred_con), axis=1)
# y_pred_1step_prime = np.concatenate((X_test[:, -1, :-1], y_pred_1step), axis=1)
# y_test_prime = np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1)
#
# y_pred_inv = scaler.inverse_transform(y_pred_prime)[:, -1]
# y_pred_inv_1step = scaler.inverse_transform(y_pred_1step_prime)[:, -1]
# y_test_inv = scaler.inverse_transform(y_test_prime)[:, -1]
#
# mae = np.mean(np.abs(y_pred_inv - y_test_inv))
# r2 = r2_score(y_pred_inv, y_test_inv)
# print('Recursive Test MAE:', mae)
# print("Recursive Test r2: ", r2)
#
# filename = f"LSTM_rec_Mv_{int(mae)}.h5"
# save_model(best_model, filename)
#
# mae_1step = np.mean(np.abs(y_pred_inv_1step - y_test_inv))
# r2_1step = r2_score(y_pred_inv_1step, y_test_inv)
# print('1 step Test MAE:', mae_1step)
# print("1 step Test r2: ", r2_1step)
#
# now = datetime.datetime.now()
# date_string = now.strftime("%Y-%m-%d %H-%M-%S")
#
# plt.figure()
# plt.plot(y_pred_inv[:horizon * days], label='recursive prediction')
# plt.plot(y_pred_inv_1step[:horizon * days], label='1 step prediction')
# plt.plot(y_test_inv[:horizon * days], label='truth')
# plt.title("LSTM_recur_Mv")
# plt.legend()
# filename = f"LSTM_recur_Mv/timePlot_{date_string}.svg"
# plt.savefig(filename)
# # plt.show()
#
# # plt.figure()
# # plt.plot(history.history['loss'], label='train')
# # plt.plot(history.history['val_loss'], label='test')
# # plt.title("lossHistory")
# # filename = f"LSTM_recur_Mv/lossHistory_{date_string}.svg"
# # plt.savefig(filename)
# # plt.legend()
# # plt.show()
#
# display = PredictionErrorDisplay.from_predictions(
#     y_true=y_test_inv,
#     y_pred=y_pred_inv,
#     kind="actual_vs_predicted",
#     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
#     line_kwargs={"color": "tab:red"},
# )
# filename = f"LSTM_recur_Mv/errorDisplay_{date_string}.svg"
# plt.savefig(filename)
# # plt.show()
