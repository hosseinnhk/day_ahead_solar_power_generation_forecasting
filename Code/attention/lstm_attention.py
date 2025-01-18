import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
# import datetime
# import os
# from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
# from keras.saving.save import save_model, load_model
# from keras.callbacks import EarlyStopping
# from sklearn.preprocessing import MinMaxScaler
# from keras.optimizers import RMSprop, SGD
# from sklearn.metrics import r2_score, PredictionErrorDisplay
# # from keras.wrappers.scikit_learn import KerasRegressor
# from sklearn.model_selection import GridSearchCV
# from keras.layers import Input, Dense, LSTM, Multiply, Activation, Lambda,Layer
# from sklearn.ensemble import RandomForestRegressor
# from keras.models import Model

# plt.rc('font', family='Times New Roman')
# plt.rcParams.update({'font.size': 14})

# if not os.path.exists("LSTM_recur_Mv"):
#     os.mkdir("LSTM_recur_Mv")

df = pd.read_csv('sf_dataset.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
data = df.pop('Power')

train_size = int(len(data) * 0.8)
train_data = np.array(data.iloc[:train_size])
test_data = np.array(data.iloc[train_size:])

plt.plot(train_data)
plt.show()

window_size = 48

# def split_sequence(sequence, n_steps):
#     X, y = list(), list()
#     for i in range(len(sequence)):
#         # find the end of this pattern
#         end_ix = i + n_steps
#         # check if we are beyond the sequence
#         if end_ix > len(sequence)-1:
#             break
#         # gather input and output parts of the pattern
#         seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
#         X.append(seq_x)
#         y.append(seq_y)
#     return array(X), array(y)

def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix])
    return np.array(X), np.array(y)


X_train, y_train = prepare_data(train_data, window_size)
X_test, y_test = prepare_data(test_data, window_size)



num_features = 1

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # Compute attention scores
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # Apply attention weights to input
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector


def build_model(T, D):
    inputs = tf.keras.Input(shape=(T, D))
    x = Bidirectional(LSTM(128, return_sequences=True)(inputs))
    x = Attention(128)(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


model = build_model(window_size, num_features)
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)
y_pred = model.predict(X_test)
y_test = y_test.flatten()
y_pred = y_pred.flatten()
print('mae:', mean_absolute_error(y_pred, y_test))
print('r2:', r2_score(y_pred, y_test))


# def create_lstm_model(learning_rate=0.00055, decay=0.01, dropOut=0.3, cells=400, dense=146, activation='tanh'):
#     inputs = tf.keras.Input(shape=(window_size, num_features))
#     x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
#     x = Attention(64)(x)
#     x = tf.keras.layers.Dense(1)(x)
#     model = tf.keras.Model(inputs=inputs, outputs=x)
#     input_layer = Input(shape=(window_size, num_features))
#     lstm_layer = LSTM(180, activation='tanh', return_sequences=True)(input_layer)
#
#
#     dropout_layer = Dropout(0.25)(lstm_layer)
#     lstm_layer2 = LSTM(180, activation='tanh', return_sequences=False)(dropout_layer)
#     dropout_layer2 = Dropout(0.25)(lstm_layer2)
#     dense_layer2 = Dense(64, activation='relu')(dropout_layer2)
#     flatten_layer1 = Flatten()(dense_layer2)
#
#     input_layer2 = Input(shape=(window_size, num_features))
#     att_layer = Attention(use_scale=True)([input_layer2, input_layer2])
#     att_layer2 = Attention(use_scale=True)([att_layer, att_layer])
#     flatten_layer2 = Flatten()(att_layer2)
#
#     merged = Concatenate()([flatten_layer1, flatten_layer2])
#
#     output_layer = Dense(1, activation='linear')(merged)
#
#     meta_model = Model(inputs=[input_layer, input_layer2], outputs=output_layer)
#     # optimizer = RMSprop(learning_rate=learning_rate, rho=0.9, epsilon=1e-08, decay=decay)
#     # optimizer = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
#     meta_model.compile(loss='mean_squared_error', optimizer='adam')
#     # meta_model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[tf.keras.metrics.MeanSquaredError()])
#     return meta_model
#
#
# early_stop = EarlyStopping(monitor='val_loss', patience=20)
# model = KerasRegressor(build_fn=create_lstm_model, epochs=60, batch_size=32, verbose=2,
#                        validation_data=([X_test, X_test], y_test), callbacks=[early_stop])

# print(model.summary())

# history = model.fit(X_train, y_train, epochs=200, batch_size=128, verbose=2,
#                     validation_data=(X_test, y_test), callbacks=[early_stop])

# history = model.fit([X_train, X_train], y_train)
#
#
# def recursive_predict(models, df, n_steps, day):
#     pred = np.zeros((n_steps * days, 1))
#     for step in range(day):
#         df_copy = df.copy()
#         for i in range(n_steps):
#             x_pred = df_copy[(i+step * 24), :, :].reshape(1, df.shape[1], df.shape[2])
#             pred[(i+step * 24), 0] = models.predict(x_pred,
#                                                     verbose=0)
#             # pred[(i + step * 24), 0] = (pred[(i+step * 24), 0] +
#             #                             lgbm_model.predict(df_copy[(i+step * 24), -1, :-1]))/2
#             df_copy[(i+step * 24)+1, -1, -1] = pred[(i+step * 24), 0]
#     return pred
#
#
# X_train_keras_preds = model.predict([X_train, X_train])
# X_test_keras_preds = model.predict([X_test, X_test])
#
# rf_model = RandomForestRegressor(n_estimators=300, random_state=42, verbose=2)
# rf_model.fit(X_train_keras_preds.reshape(-1, 1), y_train)
# y_pred_1step = rf_model.predict(X_test_keras_preds.reshape(-1, 1))
#
# y_pred_1step = np.reshape(y_pred_1step, (-1, 1))
# # y_pred = recursive_predict(rf_model, X_test_keras_preds, horizon, days)
# # y_pred_con = np.concatenate((y_pred, y_pred_1step[horizon * days:]))
#
#
# # y_pred_prime = np.concatenate((X_test[:, -1, :-1], y_pred_con), axis=1)
# y_pred_1step_prime = np.concatenate((X_test[:, -1, :-1], y_pred_1step), axis=1)
# y_test_prime = np.concatenate((X_test[:, -1, :-1], y_test.reshape(-1, 1)), axis=1)
#
# # y_pred_inv = scaler.inverse_transform(y_pred_prime)[:, -1]
# y_pred_inv_1step = scaler.inverse_transform(y_pred_1step_prime)[:, -1]
# y_test_inv = scaler.inverse_transform(y_test_prime)[:, -1]
#
# # mae = np.mean(np.abs(y_pred_inv - y_test_inv))
# # r2 = r2_score(y_pred_inv, y_test_inv)
# # print('Recursive Test MAE:', mae)
# # print("Recursive Test r2: ", r2)
#
# # filename = f"LSTM_At_rec_Mv_{int(mae)}.h5"
# # model.model.save(filename)
# # save_model(model.outputs, filename)
#
# mae_1step = np.mean(np.abs(y_pred_inv_1step - y_test_inv))
# r2_1step = r2_score(y_pred_inv_1step, y_test_inv)
# print('1 step Test MAE:', mae_1step)
# print("1 step Test r2: ", r2_1step)
#
# # now = datetime.datetime.now()
# # date_string = now.strftime("%Y-%m-%d %H-%M-%S")
#
# # plt.figure()
# # # plt.plot(y_pred_inv[:horizon * days], label='recursive prediction')
# # plt.plot(y_pred_inv_1step[:horizon * days], label='1 step prediction')
# # plt.plot(y_test_inv[:horizon * days], label='truth')
# # plt.title("LSTM_recur_Mv")
# # plt.legend()
# # filename = f"LSTM_recur_Mv/timePlot_{date_string}.svg"
# # plt.savefig(filename)
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
#
# # fig = plt.figure(figsize=(10, 10), dpi=532)
# # display = PredictionErrorDisplay.from_predictions(
# #     y_true=y_test_inv,
# #     y_pred=y_pred_inv_1step,
# #     kind="actual_vs_predicted",
# #     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
# #     line_kwargs={"color": "tab:red"},
# # )
#
# # plt.grid(linewidth=.5)
# # plt.title('LSTM 1 step')
# # plt.tight_layout()
# # filename = f"LSTM_recur_Mv/errorDisplay_{date_string}.svg"
# # plt.savefig(filename)
# # plt.show()
