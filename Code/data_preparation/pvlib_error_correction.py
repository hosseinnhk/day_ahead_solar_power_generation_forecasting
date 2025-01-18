import pandas as pd
import pvlib
import requests
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pvlib import location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS

latitude, longitude, altitude = 59.103075, 24.302820, 10    # Tallinn, Estonia
start_date = '2022-01-01'
end_date = '2023-06-30'

df = pd.read_csv('solarFarm22_23.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
df.set_index(['date', 'time'], inplace=True)
df = df[start_date:end_date]

'solar position calculation'
# date_range = pd.date_range(start='2022-01-01 00:00:00', end='2023-06-30 23:00:00', freq='H')
# solar_position_method = 'ephemeris'
# solar_position = pvlib.solarposition.get_solarposition(
#     time=date_range,
#     latitude=latitude,
#     longitude=longitude,
#     altitude=altitude,
#     pressure=pvlib.atmosphere.alt2pres(altitude),
#     method=solar_position_method)
#
# solar_pos = pd.DataFrame(index=date_range)
# solar_pos['solar_azimuth'] = solar_position['azimuth']
# solar_pos['solar_zenith'] = solar_position['zenith']
# solar_pos['solar_elevation'] = solar_position['elevation']
# solar_pos['solar_time'] = solar_position['solar_time']
# solar_pos = solar_pos.reset_index()
# solar_pos['date'] = pd.to_datetime(solar_pos['index'].dt.date)
# solar_pos['time'] = solar_pos['index'].dt.time
# solar_pos['time'] = pd.to_datetime(solar_pos['time'], format='%H:%M:%S').dt.time
# solar_pos = solar_pos.drop('index', axis=1)
# solar_pos = solar_pos.set_index(['date', 'time'])
#
# df = pd.concat([df, solar_pos], axis=1)
# df.Power[df.Power < 0] = 0

'forecast weather values'
url = 'https://archive-api.open-meteo.com/v1/archive'

params = {
    "latitude": latitude,
    "longitude": longitude,
    'start_date': start_date,
    'end_date': end_date,
    'hourly':
        'shortwave_radiation,'
        'direct_radiation,'
        'diffuse_radiation,'
        'direct_normal_irradiance,'
        'temperature_2m,'
        'dewpoint_2m,'
        'relativehumidity_2m,'
        'surface_pressure,'
        'cloudcover,'
        'cloudcover_low,'
        'cloudcover_mid,'
        'cloudcover_high,'
        'windspeed_10m,'
        'winddirection_10m,'
        'precipitation,'
        'rain,'
        'snowfall,'
        'weathercode'
        ,
    'daily':
        'weathercode,'
        'sunrise,'
        'sunset,'
        'shortwave_radiation_sum,'
        'precipitation_sum,'
        'rain_sum,'
        'snowfall_sum,'
        'precipitation_hours',
    'models': 'best_match',
    "timezone": "GMT+0",
}

response = requests.get(url, params=params)

if response.ok:
    json_data = response.json()
    dh = pd.DataFrame(json_data['hourly'])
    dh['date'] = dh['time'].str.split('T').str[0]
    dh['time'] = pd.to_datetime(dh['time']).dt.strftime('%H:%M:%S')
    dh = dh[['date', 'time', 'shortwave_radiation',
             'direct_radiation', 'diffuse_radiation',
             'direct_normal_irradiance', 'temperature_2m', 'dewpoint_2m',
             'relativehumidity_2m', 'surface_pressure', 'windspeed_10m',
             'winddirection_10m', 'cloudcover',
             'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
             'precipitation', 'rain', 'snowfall', 'weathercode']]
else:
    print(f"Request failed with status code {response.status_code}")

dh['DateTime'] = pd.to_datetime(dh['date'].astype(str) + ' ' + dh['time'].astype(str))
dh.set_index('DateTime', inplace=True)
weather = dh.drop(columns=['date', 'time'])

weather['date'] = weather.index.date
weather['time'] = weather.index.time
weather['date'] = pd.to_datetime(weather['date'])
weather['time'] = pd.to_datetime(weather['time'], format='%H:%M:%S').dt.time
weather.set_index(['date', 'time'], inplace=True)
df = pd.concat([df, weather], axis=1)
df = df.dropna(axis=0)
# df['cloud_radiation'] = df['direct_radiation'] * (100-df['cloudcover_low'])
# df.to_csv('farm1.csv')

'pv simulation'
tz = 'Etc/GMT0'
site = location.Location(latitude, longitude, tz=tz)
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
parameter_list = cec_modules.keys()
module_parameters = cec_modules['Amerisolar_Worldwide_Energy_and_Manufacturing_USA_Co___Ltd_AS_6M_345W']  #
inverter_parameters = cec_inverters['Yaskawa_Solectria_Solar__PVI_50_kW_240']
temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = PVSystem(surface_tilt=30, surface_azimuth=165,
                  module_parameters=module_parameters,
                  inverter_parameters=inverter_parameters,
                  temperature_model_parameters=temperature_model_parameters,
                  modules_per_string=70)

mc = ModelChain(system, site, aoi_model='no_loss')

date_range = pd.date_range(start='2022-01-01 00:00:00', end='2023-06-30 23:00:00', freq='H')
temperature = df[start_date:end_date].temperature_2m.values
wind_speed = df[start_date:end_date].windspeed_10m.values
dni = df[start_date:end_date].direct_normal_irradiance.values
dhi = df[start_date:end_date].diffuse_radiation.values
ghi = df[start_date:end_date].shortwave_radiation.values
real_power = df[start_date:end_date].Power.values
precipitation = df[start_date:end_date].precipitation.values


def objective(ghi_ind, dni_ind, dhi_ind, temp_air_ind, wind_speed_ind, precipitable_ind):
    param = {
        'ghi_ind': ghi_ind,
        'dni_ind': dni_ind,
        'dhi_ind': dhi_ind,
        'temp_air_ind': temp_air_ind,
        'wind_speed_ind': wind_speed_ind,
        'precipitable_ind': precipitable_ind,
    }

    weather_df = pd.DataFrame(index=date_range, data={
        'ghi': param['ghi_ind']*ghi,  # Global horizontal irradiance
        'dni': param['dni_ind']*dni,  # Direct normal irradiance
        'dhi': param['dhi_ind']*dhi,  # Diffuse horizontal irradiance
        'temp_air': param['temp_air_ind']*temperature,  # Ambient temperature in degC
        'wind_speed': param['wind_speed_ind']*wind_speed,  # Wind speed in m/s
        'precipitable_water': param['precipitable_ind']*precipitation  # millimeters #
    })

    mc.run_model(weather_df)
    mse = mean_squared_error(mc.results.ac, real_power)
    return -mse


param_ranges = {
        'ghi_ind': (0.1, 5),
        'dni_ind': (0.1, 2),
        'dhi_ind': (0.1, 2),
        'temp_air_ind': (0.1, 1.5),
        'wind_speed_ind': (0.1, 2),
        'precipitable_ind': (0.01, 1),
}

optimizer = BayesianOptimization(f=objective, pbounds=param_ranges, random_state=42)
optimizer.maximize(init_points=10, n_iter=500)

best_params = optimizer.max['params']
best_mse = -optimizer.max['target']

print("Best Hyperparameters:")
for key, value in best_params.items():
    print(f"{key}: {value}")
print(f"Best MSE: {best_mse}")
