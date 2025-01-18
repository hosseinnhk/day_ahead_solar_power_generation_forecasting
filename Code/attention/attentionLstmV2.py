import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from keras.layers import LSTM, Bidirectional

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv('sf_dataset.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.pop('Power')

train_df = df[df.index.get_level_values('date') < '2023-02-01']
test_df = df[df.index.get_level_values('date') >= '2023-02-01']

train_df = train_df.values.reshape(-1, 1)
test_df = test_df.values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_df)

train_df = scaler.transform(train_df)
test_df = scaler.transform(test_df)

window_size = 168


def prepare_data(sequence, n_steps):
    X, y = [], []
    for i in range(0, len(sequence), 24):
        end_ix = i + n_steps
        if end_ix+24 > len(sequence) - 1:
            break
        X.append(sequence[i:end_ix])
        y.append(sequence[end_ix:end_ix+24])
    return np.array(X), np.array(y)


X_train, y_train = prepare_data(train_df, window_size)

X_test, y_test = prepare_data(test_df, window_size)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, inputs):
        score = tf.nn.tanh(self.W(inputs))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector


def build_model(T, D):
    inputs = tf.keras.Input(shape=(T, D))
    x = Bidirectional(LSTM(400, return_sequences=True))(inputs)
    x = Attention(400)(x)
    x = tf.keras.layers.Dense(24)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model


model = build_model(window_size, 1)
model.compile(optimizer="adam", loss="mse")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
y_pred = model.predict(X_test)
y_test = np.squeeze(y_test)

y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)


def mean_absolute_percentage(y_true, y_pred):
    # y_true, y_pred = list(map(float, y_true)), list(map(float, y_pred))
    # assert len(y_true) == len(y_pred), "Input lists must have the same length"
    ymean = np.mean(y_true)
    ape = [abs(y_true[i] - y_pred[i]) for i in range(len(y_true))]

    # Return the mean of the absolute percentage errors
    mape = (sum(ape) / len(y_true)) * 100
    mape = mape / ymean

    return mape


print('test performance:')
print('\tnRmse:', mean_squared_error(y_test, y_pred, squared=False) / np.max(y_test))
print('\tMAPE:', np.mean(mean_absolute_percentage(y_test, y_pred)))
print('\tnMAE:', mean_absolute_error(y_pred, y_test)/ np.max(y_test))
print('\tR2:', r2_score(y_pred, y_test))

# Plotting training and validation loss curves
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(y_test[:7*24], '--', label='actual')
plt.plot(y_pred[:7*24], label='actual')
plt.legend()
plt.show()