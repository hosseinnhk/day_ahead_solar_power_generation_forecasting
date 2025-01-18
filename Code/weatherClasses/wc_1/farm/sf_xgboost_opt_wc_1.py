import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, PredictionErrorDisplay
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from optuna.samplers import TPESampler

warnings.filterwarnings('ignore')

df = pd.read_csv('lightgbm_pred_753.csv')
df = df.reset_index(drop=True)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop(['date', 'time'], axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[753:, -1]
x = scaled_df[753:, -2]

train_size = int(len(x) * 0.80)

X_train, X_test = x[0:train_size], x[train_size:]
Y_train, Y_test = y[0:train_size], y[train_size:]

kf = KFold(n_splits=10, shuffle=True, random_state=42)


def objective(trial):

    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 6, 15),  # Extremely prone to over fitting!
        'n_estimators': trial.suggest_int('n_estimators', 400, 4000, 400),  # Extremely prone to over fitting!
        'eta': trial.suggest_float('eta', 0.007, 0.013),  # Most important parameter.
        'subsample': trial.suggest_discrete_uniform('subsample', 0.2, 0.9, 0.1),
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.2, 0.9, 0.1),
        'colsample_bylevel': trial.suggest_discrete_uniform('colsample_bylevel', 0.2, 0.9, 0.1),
        'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-4, 1e4),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1e4),  # L2 regularization
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1e4),  # L1 regularization
        'gamma': trial.suggest_loguniform('gamma', 1e-4, 1e4),
    }

    mse_scores = []

    for train_index, val_index in kf.split(X_train):
        x_train, x_val = X_train[train_index].reshape(-1, 1), X_train[val_index].reshape(-1, 1)
        y_train, y_val = Y_train[train_index], Y_train[val_index]
        reg = xgb.XGBRegressor(
            objective='reg:squarederror',
            predictor='cpu_predictor',
            n_jobs=-1,
            random_state=42,
            **param_grid
        )
        reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', verbose=False)
        mse_scores.append(mean_squared_error(y_val, reg.predict(x_val)))
    return np.mean(mse_scores)


train_time = 60 * 10
study = optuna.create_study(direction='minimize', sampler=TPESampler(), study_name='XGBRegressor')
study.optimize(objective, timeout=train_time)

print('Number of finished trials: ', len(study.trials), '\n_____________________________________')
print('Best trial:')
trial = study.best_trial

print('\tmse value: {}'.format(trial.value))
print('\tParams: ')
for key, value in trial.params.items():
    print('\t\t{}: {}'.format(key, value))

# optuna.visualization.matplotlib.plot_optimization_history(study)
# # plt.savefig('optimization_history.svg', format='svg')
# plt.show()
#
# optuna.visualization.matplotlib.plot_parallel_coordinate(study)
# plt.show()
#
# optuna.visualization.matplotlib.plot_param_importances(study)
# plt.show()

# optuna.visualization.matplotlib.plot_intermediate_values(study)
# plt.show()

xgb_params = trial.params

mse_scores = []
r2_scores = []

for train_index, val_index in kf.split(X_train):
    x_train, x_val = X_train[train_index].reshape(-1, 1), X_train[val_index].reshape(-1, 1)
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        predictor='cpu_predictor',
        n_jobs=-1,
        random_state=42,
        **xgb_params
    )
    reg.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='rmse', verbose=False)
    y_pred = reg.predict(x_val)
    mse_scores.append(mean_squared_error(y_val, y_pred))
    r2_scores.append(r2_score(y_val, y_pred))
print('\n_____________________________________')
print('train:')
print('mse score:', np.mean(mse_scores))
print('r2 score:', np.mean(r2_scores), '\n_____________________________________')

pred_test = reg.predict(X_test.reshape(-1, 1))

print('test:')
print('nmse score:', mean_squared_error(Y_test, pred_test) / np.max(Y_test))
print('r2 score:', r2_score(Y_test, pred_test))

# display = PredictionErrorDisplay.from_predictions(
#     y_true=y_test,
#     y_pred=model.predict(X_test),
#     kind="actual_vs_predicted",
#     scatter_kwargs={"alpha": 0.2, "color": "tab:blue"},
#     line_kwargs={"color": "tab:red"},
# )
# plt.show()

