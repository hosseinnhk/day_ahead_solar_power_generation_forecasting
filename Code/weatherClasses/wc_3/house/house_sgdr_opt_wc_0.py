import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDRegressor
from bayes_opt import BayesianOptimization

# Best Hyperparameters:
# alpha: 0.0
# eta0: 1.0
# max_iter: 1545
# power_t: 0.72
# tol: 3.355
# Best MSE: 0.016

# Best Hyperparameters:
# alpha: 0.0
# eta0: 0.187
# max_iter: 1742
# power_t: 0.47
# tol: 3.35
# Best MSE: 0.017

df = pd.read_csv('house_base_regs_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time', 'Unnamed: 0'], axis=1)

y = df.actual.values
x = df.drop(['actual'], axis=1)

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]


def objective(eta0, power_t, alpha, max_iter):  # , l1_ratio , tol
    params = {
        'penalty': 'elasticnet',
        'learning_rate': 'invscaling',  # optimal
        'eta0': eta0,
        'power_t': power_t,
        'alpha': alpha,
        # 'l1_ration': l1_ratio,
        'max_iter': int(max_iter),
        # 'tol': tol,
    }

    mse_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, test_index in kf.split(X_train):
        x_train, x_val = X_train.loc[train_index], X_train.loc[test_index]
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
    'alpha': (0, 1),
    # 'l1_ration': (0, 1),
    'max_iter': (100, 2000),
    # 'tol': (0, 0.001),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=20, n_iter=300)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")
