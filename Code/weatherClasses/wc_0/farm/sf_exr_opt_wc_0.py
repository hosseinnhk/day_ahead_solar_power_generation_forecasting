import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor as exr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

# Best Hyperparameters:
# max_depth: 36
# max_features: 1.0
# min_samples_leaf: 1.0
# min_samples_split: 10.0
# n_estimators: 162
# Best MSE: 0.003590504131614311

# Best Hyperparameters:
# max_depth: 37
# max_features: 16
# max_leaf_nodes: 14
# min_impurity_decrease: 0.0
# min_samples_leaf: 34
# min_samples_split: 10.0
# n_estimators: 2077
# Best MSE: 0.0066301267058321315

df = pd.read_csv('sf_wc_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)
df = df.drop(['weathercode', 'snowfall'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.60)

x = x[0:train_size]
y = y[0:train_size]


def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features,
              max_leaf_nodes, min_impurity_decrease,
              # min_weight_fraction_leaf, max_samples
              ):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        "min_samples_split": int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        # 'min_weight_fraction_leaf': min_weight_fraction_leaf,
        'max_features': int(max_features),
        'criterion': 'absolute_error',
        'max_leaf_nodes': int(max_leaf_nodes),
        'min_impurity_decrease': min_impurity_decrease,
        # 'max_samples': max_samples,
        'n_jobs': -1,
        'random_state': 42,
        'verbose': 0,
    }

    mse_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    model = exr(**params)
    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    average_mse = np.mean(mse_scores)
    return -average_mse


param_ranges = {
    'n_estimators': (100, 3000),
    'max_depth': (5, 50),
    "min_samples_split": (1, 10),
    'min_samples_leaf': (1, 100),
    # 'min_weight_fraction_leaf': (),
    'max_features': (1, 20),
    'max_leaf_nodes': (2, 30),
    'min_impurity_decrease': (0, 0.05),
    # 'max_samples': (),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")
print(f"Best MSE: {best_mse}")


