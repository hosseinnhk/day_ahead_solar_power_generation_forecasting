import shap
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.model_selection import train_test_split
sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=18)

df = pd.read_csv('farm_dataset.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=True)


# print(df.head())
# df = df.drop([
#     'weathercode', 'cloudcover', 'direct_normal_irradiance', 'direct_radiation', 'cloudcover_high',
#     'solar_time', 'precipitation', 'rain', 'snowfall', 'Lag_25', 'Lag_49', 'diffuse_radiation',
#     'surface_pressure', 'winddirection_10m', 'windspeed_10m', 'Lag_47', 'relativehumidity_2m',
#     # 'showers', 'visibility',
#     #  'snow_depth',
#     # 'precipitation_probability', 'windspeed_10m',
#     # 'uv_index_clear_sky',
#     # 'shortwave_radiation_instant', 'terrestrial_radiation',
#     # 'solar_zenith', 'relativehumidity_2m', 'direct_radiation_instant',
#     # 'temperature_2m', 'winddirection_10m', 'solar_azimuth',
#     # 'dewpoint_2m', 'solar_elevation', 'terrestrial_radiation_instant',
# ], axis=1)
# df = df[df.shortwave_radiation > 0]  # only day values
# X_raw = df.drop('Power', axis=1)

# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_df = scaler.fit_transform(df.values)
df = df.drop('pvsim', axis=1)
y = df['Power']
X = df.drop('Power', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# y = scaled_df[:, 0]
# x = scaled_df[:, 1:]

# train_size = int(len(x) * 0.80)
# train_size = train_size - train_size % 24
#
# X_train, X_test = x[0:train_size, :], x[train_size:, :]
# Y_train, Y_test = y[0:train_size], y[train_size:]

params = {
    "loss": "squared_error",
    "learning_rate": 0.01,
    "n_estimators": 433,
    # "subsample": 1,
    # "criterion": "friedman_mse",
    # "min_samples_split": 2,
    # "min_samples_leaf": 1,
    # "min_weight_fraction_leaf": 0.0,
    # "max_depth": 3,
    # "min_impurity_decrease": 0.0,
    # "init": None,
    "max_features": 9,
    # "alpha": 0.1,
    # "verbose": 0,
    # "max_leaf_nodes": None,
    # "warm_start": True,
    # "validation_fraction": 0.1,
    # "n_iter_no_change": None,
    # "tol": 0.0001,
    # "ccp_alpha": 0.0
}

# reg = GradientBoostingRegressor(**params)
reg = RandomForestRegressor()
reg.fit(X_train, y_train)
explainer = shap.TreeExplainer(reg)  #Explainer(reg)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, show=False)
# instance_index = 0
# shap.force_plot(explainer.expected_value, shap_values[instance_index], X_test[instance_index])
# ax.set_xlabel('', fontsize=14)
# ax.set_ylabel('', fontsize=14)
# plt.title("Customized SHAP Summary Plot", fontsize=16)  # Set the title and its fontsize
# plt.xlabel("", fontsize=16)  # Set the x-axis label and its fontsize
# plt.ylabel("Feature", fontsize=16)  # Set the y-axis label and its fontsize
plt.tick_params(axis="both", labelsize=14)

# plt.savefig("shap_summary_plot2.svg")
plt.show()

shap.plots.bar(shap_values)
# mse = mean_squared_error(y_test, reg.predict(X_test))
# print("The mean squared error (MSE) on test set: {:.4f}".format(mse))

# test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
# for i, y_pred in enumerate(reg.predict(X_test)):
#     test_score[i] = mean_squared_error(y_test, y_pred)

# fig = plt.figure(figsize=(6, 6))
# plt.subplot(1, 1, 1)
# plt.title("Deviance")
# plt.plot(
#     np.arange(params["n_estimators"]) + 1,
#     reg.train_score_,
#     "b-",
#     label="Training Set Deviance",
# )
# plt.plot(
#     np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
# )
# plt.legend(loc="upper right")
# plt.xlabel("Boosting Iterations")
# plt.ylabel("Deviance")
# fig.tight_layout()
# plt.show()
##############
# feature_importance = reg.feature_importances_
#
# sorted_idx = np.argsort(feature_importance)[-14:-1]
# # print(np.array(X_raw.columns)[sorted_idx])
#
# # pos = np.arange(sorted_idx.shape[0]) + 0.5
# # fig = plt.figure(figsize=(12, 6))
# # # plt.subplot(1, 2, 1)
# # plt.barh(pos, feature_importance[sorted_idx], align="center", color='black')
# # plt.yticks(pos, np.array(X_raw.columns)[sorted_idx], fontsize=16, rotation=30)
# # # plt.title("Feature Importance", fontsize=20)
# # plt.xticks(fontsize=16)  # Adjust the fontsize as desired
# # plt.tight_layout()
# # sns.despine()
# # plt.savefig('feature_importance.svg')
# # plt.show()
#
# result = permutation_importance(
#     reg, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
# )
# sorted_idx = result.importances_mean.argsort()[:-1]
# # print(sorted_idx)
# plt.subplot(1, 2, 2)
# fig = plt.figure(figsize=(12, 6))
# plt.boxplot(
#     result.importances[sorted_idx].T,
#     vert=False,
#     labels=np.array(X_raw.columns)[sorted_idx],
# )
# # plt.title("Permutation Importance (test set)", fontsize=20)
# plt.xticks(fontsize=16)  # Adjust the fontsize as desired
# plt.yticks(fontsize=18)  # Adjust the fontsize as desired
# fig.tight_layout()
# sns.despine()
# # plt.savefig('permutation_importance.svg')
# plt.show()

