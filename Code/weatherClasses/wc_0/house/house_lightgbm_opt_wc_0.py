import numpy as np
import pandas as pd
import warnings
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

# Best Hyperparameters:
# learning_rate: 0.08
# max_depth: 8
# min_data_in_leaf: 2
# n_estimators: 140
# num_iterations: 1825
# num_leaves: 97
# Best MSE: 0.0105

# Best Hyperparameters:
# feature_fraction: 0.93
# learning_rate: 0.097
# max_depth: 4
# min_data_in_leaf: 9
# n_estimators: 142
# num_iterations: 713
# num_leaves: 91
# reg_alpha: 0.56
# reg_lambda: 7.7


df = pd.read_csv('house_wc_0.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)
df = df.drop(['weathercode', 'snowfall',
              'is_day', 'shortwave_radiation_instant', 'direct_radiation_instant',
              'diffuse_radiation_instant',
              'direct_normal_irradiance_instant', 'terrestrial_radiation',
              'terrestrial_radiation_instant', 'uv_index_clear_sky', 'uv_index',
              'visibility', 'precipitation_probability', 'showers',
              'snow_depth'
              ], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

train_size = int(len(x) * 0.80)

x = x[0:train_size]
y = y[0:train_size]

'''uncomment to apply PCA'''
# pca = PCA(n_components=7)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cumsum = np.cumsum(explained_variance_ratio)
# dim = np.argmax(cumsum >= 0.95) + 1
# print(dim)
# x = pca.transform(x)


def objective(
        learning_rate,
        num_iterations,
        max_depth,
        n_estimators,
        num_leaves,
        feature_fraction,
        # min_child_samples,
        reg_alpha,
        reg_lambda,
        # min_child_weight,
        min_data_in_leaf
):
    params = {
        'task': 'train',
        'boosting_type': 'goss',
        'objective': 'mape',
        'metric': 'rmse',
        'force_col_wise': True,
        'num_leaves': int(num_leaves),
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        # 'min_child_samples': int(min_child_samples),
        'min_data_in_leaf': int(min_data_in_leaf),
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        "num_iterations": int(num_iterations),
        # 'min_child_weight': min_child_weight,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'n_jobs': -1,
        'class_weight': None,
        'min_split_gain': 0.0,
        'random_state': 42,
        'subsample_freq': 0,
        'verbose': 0,
        'early_stopping_round': 10,
        'feature_pre_filter': True,
        'max_bin': 1000,
    }

    mse_scores = []
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(x):
        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_data = lgb.Dataset(x_train, label=y_train)
        val_data = lgb.Dataset(x_val, label=y_val)
        model = lgb.train(params, train_data, valid_sets=[val_data])
        y_pred = model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)
    average_mse = np.mean(mse_scores)
    return -average_mse


param_ranges = {
    'num_leaves': (80, 150),
    'learning_rate': (0.05, 0.1),
    'feature_fraction': (0.9, 1),
    # 'min_child_samples': (0, 20),
    # 'min_child_weight': (0, 20),
    'n_estimators': (50, 200),
    'max_depth': (3, 10),
    "num_iterations": (600, 1000),
    'reg_alpha': (0, 1),
    'reg_lambda': (7, 10),
    'min_data_in_leaf': (40, 100),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=10, n_iter=50)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")

print(f"Best MSE: {best_mse}")
