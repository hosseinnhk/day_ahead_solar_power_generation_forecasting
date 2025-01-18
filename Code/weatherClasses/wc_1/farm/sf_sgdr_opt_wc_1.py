import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from bayes_opt import BayesianOptimization

# Best Hyperparameters:
# alpha: 0.0022309718685575852
# eta0: 0.2771626471945892
# max_iter: 7968.618756997194
# power_t: 0.5118803158701186
# tol: 3.5646791422077992
# Best MSE: 0.005113552274942756

# Best Hyperparameters:
# alpha: 0.0
# eta0: 0.001
# max_iter: 1073.8102683509026
# power_t: 0.0
# tol: 0.0
# Best MSE: 0.005410185296800934

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

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

# only with lightgbm:
# Best Hyperparameters:
# alpha: 0.00074
# eta0: 0.46
# max_iter: 1100
# power_t: 0.34
# tol: 1.48e-05
# Best MSE: 0.01321


def objective(eta0, power_t, alpha, max_iter, tol):  # l1_ratio,
    params = {
        'penalty': 'elasticnet',
        'learning_rate': 'invscaling',  # optimal
        'eta0': eta0,
        'power_t': power_t,
        'alpha': alpha,
        # 'l1_ration': l1_ratio,
        'max_iter': int(max_iter),
        'tol': tol,
    }

    mse_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
        x_train, x_val = X_train[train_index], X_train[test_index]
        y_train, y_val = Y_train[train_index], Y_train[test_index]
        model = SGDRegressor(random_state=42, **params)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    average_mse = np.mean(mse_scores)
    return -average_mse


param_ranges = {
    'eta0': (0.001, 1),
    'power_t': (0, 1),
    'alpha': (0, 0.005),
    # 'l1_ration': (0, 1),
    'max_iter': (50, 2000),
    'tol': (0, 0.001),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=20, n_iter=400)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")

