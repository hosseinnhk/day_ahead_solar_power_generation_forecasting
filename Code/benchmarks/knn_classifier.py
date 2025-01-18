import numpy as np
import pandas as pd
import warnings
import logging
import joblib
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logging.getLogger('lightgbm').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')

file_path = \
    'D:/TalTechUniversity/solarIradiance/forecasting/nZEB_mektory/solar_farm/data_prepration/farm/datasets/sf_dataset.csv'
df = pd.read_csv(file_path, parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index(drop=False)
date = df.date.tolist()
time = df.time.tolist()
df = df.drop([
    'date',
    'time',
    # 'pvsim',
    # 'Lag_24',
    'Lag_25',
    'Lag_48',
    # 'Lag_72',

    'solar_azimuth',
    'solar_zenith',
    'solar_elevation',
    'solar_time',

    # 'shortwave_radiation',
    # 'direct_radiation',
    'diffuse_radiation',
    # 'direct_normal_irradiance',

    'weathercode',

    # 'temperature_2m',
    'dewpoint_2m',

    'relativehumidity_2m',
    # 'surface_pressure',

    # 'windspeed_10m',
    # 'winddirection_10m',

    # 'cloudcover',
    'cloudcover_low',
    'cloudcover_mid',
    'cloudcover_high',
    'cloud_radiation',

    'precipitation',
    'rain',
    'snowfall',
], axis=1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_df = scaler.fit_transform(df.values)

y = scaled_df[:, 0]
x = scaled_df[:, 1:]

'''uncomment to apply PCA'''
# pca = PCA(n_components=14)
# X_pca = pca.fit_transform(x)
# explained_variance_ratio = pca.explained_variance_ratio_
# cum_sum = np.cumsum(explained_variance_ratio)
# dim = np.argmax(cum_sum >= 0.95) + 1
# x = pca.transform(x)

train_size = int(len(x) * 0.60)

X_train, X_test = x[0:train_size, :], x[train_size:, :]
Y_train, Y_test = y[0:train_size], y[train_size:]


k = 5
knn_classifier = KNeighborsRegressor(n_neighbors=k)
knn_classifier.fit(X_train, Y_train)
y_pred_classification = knn_classifier.predict(X_test)

class_subsets = [X_test[y_pred_classification == class_num] for class_num in np.unique(y_pred_classification)]

# Step 3: Train Regression Models for each class
regression_models = []
for subset in class_subsets:
    # Assuming the target values for each class are in the last column of the subset
    y_regression = subset[:, -1]
    X_regression = subset[:, :-1]  # Features for the regression
    regression_model = lgb.LGBMRegressor()
    regression_model.fit(X_regression, y_regression)
    regression_models.append(regression_model)

# Step 4: Make Predictions with Regression Models for new samples
# Assuming you have new_samples with shape (n_samples, n_features)
# for sample in new_samples:
#     class_prediction = knn_classifier.predict(sample.reshape(1, -1))
#     regression_model = regression_models[int(class_prediction)]
#     regression_prediction = regression_model.predict(sample.reshape(1, -1))
#     print(f"Class Prediction: {class_prediction}, Regression Prediction: {regression_prediction}")