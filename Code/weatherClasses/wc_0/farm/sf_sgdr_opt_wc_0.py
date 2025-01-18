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

df = pd.read_csv('lightgbm_1.csv')
df = df.reset_index(drop=True)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
df = df.sample(frac=1)  # shuffle dataset
print(df.columns)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, -3]
x = scaled_df[:, -2]


train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

# Best Hyperparameters  base_regs01.csv:
# alpha: 0.0004792432848518318
# eta0: 0.07226526670090627
# max_iter: 6533.644546958956
# power_t: 0.3895980085602906
# tol: 0.009532101379932208
# Best MSE: 0.0029323907570982647

# alpha: 0.0013534747207410254
# eta0: 0.062145203690378924
# max_iter: 5574.283464326584
# power_t: 0.36002864963950976
# tol: 0.20112551543779555
# Best MSE: 0.002396100873325333

# alpha: 0.0
# eta0: 0.5168593366390694
# max_iter: 7175.490092252022
# power_t: 0.6279012363472644
# tol: 1.0
# Best MSE: 0.0023874985737890667

# Best Hyperparameters: sf_wc0
# alpha: 0.0
# eta0: 0.6946649349323331
# max_iter: 7604.137372378269
# power_t: 0.22227801975995978
# tol: 3.101409683190671
# Best MSE: 0.001516494667892675

# Best Hyperparameters:
# alpha: 0.00223
# eta0: 0.277
# max_iter: 7968
# power_t: 0.512
# tol: 3.56
# Best MSE: 0.004867482341727897

# only with lightgbm:


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
        model.fit(x_train.reshape(-1, 1), y_train)
        y_pred = model.predict(x_val.reshape(-1, 1))
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
optimizer.maximize(init_points=20, n_iter=100)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")
