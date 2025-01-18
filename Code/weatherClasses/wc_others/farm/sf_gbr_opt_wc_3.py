import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as gbr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
import warnings

warnings.filterwarnings('ignore')

# Best Hyperparameters:
# learning_rate: 0.007211377105261663
# max_depth: 49.267713379968086
# max_features: 0.230792818874879
# min_samples_leaf: 8.251168739915268
# min_samples_split: 19.82910203897579
# n_estimators: 2645.6962038702773
# subsample: 0.8855500712474639
# Best MSE: 0.00338

# Best Hyperparameters:
# learning_rate: 0.038
# max_depth: 4
# max_features: 8
# min_samples_leaf: 5.
# min_samples_split: 14
# n_estimators: 791
# subsample: 0.68
# Best MSE: 0.00355

df = pd.read_csv('sf_wc_1.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)
df = df.drop(['weathercode', 'snowfall'], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.60)

x = x[0:train_size]
y = y[0:train_size]


def objective(learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, subsample,
              # min_weight_fraction_leaf,  min_impturity_decrease, max_leaf_nodes
              ):
    params = {
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        "min_samples_split": int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf),
        # 'min_weight_fraction_leaf': int(min_weight_fraction_leaf),
        'max_features': int(max_features),
        'subsample': subsample,
    }

    mse_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = gbr(random_state=42, **params)
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
    'learning_rate': (0.001, 0.1),
    'n_estimators': (100, 3000),
    'max_depth': (3, 50),
    "min_samples_split": (1, 20),
    'min_samples_leaf': (1, 100),
    # 'min_weight_fraction_leaf': ,
    'max_features': (5, 20),
    'subsample': (0.1, 1),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")
