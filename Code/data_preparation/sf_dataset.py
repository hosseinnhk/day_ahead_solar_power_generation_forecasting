import pandas as pd
import pvlib
import requests
import numpy as np
import matplotlib.pyplot as plt
from pvlib import location
from pvlib.pvsystem import PVSystem
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=20)
pd.set_option('display.max_columns', None)

latitude, longitude, altitude = 59.103075, 24.302820, 45    # Tallinn, Estonia
start_date = '2022-01-01'
end_date = '2023-06-30'

df = pd.read_csv('datasets/solarFarm22_23.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df.reset_index()
df['date'] = pd.to_datetime(df['date'])
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.time
df.set_index(['date', 'time'], inplace=True)
df = df[start_date:end_date]

'solar position calculation'
date_range = pd.date_range(start='2022-01-01 00:00:00', end='2023-06-30 23:00:00', freq='H')
solar_position_method = 'ephemeris'
solar_position = pvlib.solarposition.get_solarposition(
    time=date_range,
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    pressure=pvlib.atmosphere.alt2pres(altitude),
    method=solar_position_method)

solar_pos = pd.DataFrame(index=date_range)
solar_pos['solar_azimuth'] = solar_position['azimuth']
solar_pos['solar_zenith'] = solar_position['zenith']
solar_pos['solar_elevation'] = solar_position['elevation']
solar_pos['solar_time'] = solar_position['solar_time']
solar_pos = solar_pos.reset_index()
solar_pos['date'] = pd.to_datetime(solar_pos['index'].dt.date)
solar_pos['time'] = solar_pos['index'].dt.time
solar_pos['time'] = pd.to_datetime(solar_pos['time'], format='%H:%M:%S').dt.time
solar_pos = solar_pos.drop('index', axis=1)
solar_pos = solar_pos.set_index(['date', 'time'])

df = pd.concat([df, solar_pos], axis=1)
df.Power[df.Power < 0] = 0

'forecast weather values'
url = 'https://archive-api.open-meteo.com/v1/archive' # "https://api.open-meteo.com/v1/forecast"

# params = {
#     "latitude": latitude,
#     "longitude": longitude,
#     "hourly":
#         "is_day,"
#         "shortwave_radiation,"
#         "direct_radiation,"
#         "diffuse_radiation,"
#         "direct_normal_irradiance,"
#         "terrestrial_radiation,"
#         "uv_index,"
#         "uv_index_clear_sky,"
#         "shortwave_radiation_instant,"
#         "direct_radiation_instant,"
#         "diffuse_radiation_instant,"
#         "direct_normal_irradiance_instant,"
#         "terrestrial_radiation_instant,"
#         "temperature_2m,"
#         "dewpoint_2m,"
#         "relativehumidity_2m,"
#         "surface_pressure,"
#         "windspeed_10m,"
#         "winddirection_10m,"
#         "cloudcover,"
#         "cloudcover_low,"
#         "cloudcover_mid,"
#         "cloudcover_high,"
#         "cloudcover_400hPa,"
#         "visibility,"
#         "precipitation_probability,"
#         "precipitation,"
#         "rain,"
#         "showers,"
#         "snowfall,"
#         "snow_depth,"
#         "weathercode,",
#     "models": "best_match",
#     "daily": "sunrise,sunset",
#     "forecast_days": "1",
#     "start_date": start_date,
#     "end_date": end_date,
#     "timezone": "Europe/Moscow"
# }

# response = requests.get(url, params=params)
#
# if response.ok:
#     data_dict = response.json()
#     hourly_data = data_dict['hourly']
#     daily_data = data_dict['daily']
#     weather = pd.DataFrame(hourly_data)
#     weather['date'] = weather['time'].str.split('T').str[0]
#     weather['time'] = pd.to_datetime(weather['time']).dt.strftime('%H:%M:%S')
#     weather = weather[['date', 'time', 'is_day', 'shortwave_radiation', 'shortwave_radiation_instant',
#                        'direct_radiation', 'direct_radiation_instant', 'diffuse_radiation',
#                        'diffuse_radiation_instant', 'direct_normal_irradiance', 'direct_normal_irradiance_instant',
#                        'terrestrial_radiation', 'terrestrial_radiation_instant', 'uv_index_clear_sky', 'uv_index',
#                        'temperature_2m', 'dewpoint_2m', 'relativehumidity_2m', 'surface_pressure', 'windspeed_10m',
#                        'winddirection_10m', 'cloudcover',  'cloudcover_low', 'cloudcover_mid',
#                        'cloudcover_high', 'visibility', 'precipitation_probability', 'precipitation', 'rain', 'showers',
#                        'snowfall', 'snow_depth', 'weathercode']] #'cloudcover_400hPa',
# else:
#     print(f"Request failed with status code {response.status_code}")

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
df['cloud_radiation'] = df['direct_radiation'] * (100-df['cloudcover_low'])
df.to_csv('datasets/farm1.csv')
print(df.dtypes)


'pv simulation'
tz = 'Etc/GMT0'
site = location.Location(latitude, longitude, tz=tz)
cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
parameter_list = cec_modules.keys()
module_parameters = cec_modules['Amerisolar_Worldwide_Energy_and_Manufacturing_USA_Co___Ltd_AS_6M_345W']  #

# module_parameters = cec_modules['alfasolar_alfasolar_M6L60_240']

# for i in parameter_list:
#     print(i)

# inverter_parameters = cec_inverters['ABB__PVI_3_0_OUTD_S_US_A__240V_']
inverter_parameters = cec_inverters['Yaskawa_Solectria_Solar__PVI_50_kW_240']

# parameter_list = cec_inverters.keys()
# for i in parameter_list:
#     print(i)

temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = PVSystem(surface_tilt=30, surface_azimuth=165,
                  module_parameters=module_parameters,
                  inverter_parameters=inverter_parameters,
                  temperature_model_parameters=temperature_model_parameters,
                  modules_per_string=70)

mc = ModelChain(system, site, aoi_model='no_loss')

temperature = df[start_date:end_date].temperature_2m.values
wind_speed = df[start_date:end_date].windspeed_10m.values
dni = df[start_date:end_date].direct_normal_irradiance.values
dhi = df[start_date:end_date].diffuse_radiation.values
ghi = df[start_date:end_date].shortwave_radiation.values
real_power = df[start_date:end_date].Power.values
dewpoint_2m = df[start_date:end_date].dewpoint_2m.values
precipitation = df[start_date:end_date].precipitation.values

weather = pd.DataFrame(index=date_range, data={
    'ghi': ghi,  # Global horizontal irradiance
    'dni': dni,  # Direct normal irradiance
    'dhi': dhi,  # Diffuse horizontal irradiance
    'temp_air': temperature,  # Ambient temperature in degC
    'wind_speed': wind_speed,  # Wind speed in m/s
    'precipitable_water': precipitation  # millimeters #
})

mc.run_model(weather)

pvsim = pd.DataFrame(mc.results.ac, columns=['pvsim'])
pvsim = pvsim.reset_index(drop=False)
pvsim['index'] = pd.to_datetime(pvsim['index'])
pvsim.set_index('index', inplace=True)
pvsim['date'] = pvsim.index.date
pvsim['time'] = pvsim.index.time
pvsim['date'] = pd.to_datetime(pvsim['date'])
pvsim['time'] = pd.to_datetime(pvsim['time'], format='%H:%M:%S').dt.time
pvsim.set_index(['date', 'time'], inplace=True)
final_dt = pd.concat([df, pvsim], axis=1)
final_dt.pvsim.values[final_dt.pvsim < 0] = 0

print(np.corrcoef(real_power, pvsim.pvsim.values))
print(r2_score(real_power, pvsim.pvsim.values))
print(mean_absolute_error(pvsim.pvsim.values, real_power)/np.max(real_power))

max_lag = 168
autocorr_values = final_dt['Power'].autocorr(lag=max_lag)
plot_acf(final_dt['Power'], lags=max_lag)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Plot')
plt.show()

shift_num = 73
dummy = pd.DataFrame()
for i in range(24, shift_num):
    final_dt[f'Lag_{i}'] = final_dt.Power.shift(i).bfill()
    dummy[f'{i}'] = final_dt.Power.shift(i).bfill()
    if final_dt[f'Lag_{i}'].corr(final_dt['Power']) < 0.75:
        final_dt = final_dt.drop([f'Lag_{i}'], axis=1)

labels = []
labels = range(24, 97)

correlation_factors = dummy[[f'{i}' for i in range(24, shift_num)]].corrwith(final_dt['Power'])
correlation_factors_sorted = correlation_factors.sort_values()

plt.figure(figsize=(12, 6))
correlation_factors.plot(kind='bar', color='#155a79', alpha=.8)
plt.ylabel('Correlation Ratio', fontsize=20)
plt.xticks(rotation=90, fontsize=14)
plt.xlabel('Lags (h)', fontsize=20)
plt.axhline(y=0.75, color='#e64f04', linestyle='--')
plt.text(4, 0.76, 'Threshold: 0.75', color='#e64f04', fontsize=17)
sns.despine()
plt.tight_layout()
# plt.savefig('autoCorrelations4.svg')
plt.show()

# plt.figure(figsize=(10, 8))
# correlation_factors_sorted[-11:].plot(kind='barh', color='steelblue')
# plt.xlabel('Correlation Factor', fontsize=20)
# plt.grid(axis='x', linestyle='--')
# plt.axvline(x=0.70, color='red', linestyle='--')
# plt.tight_layout()
# plt.savefig('correlations2.svg')
# plt.show()


# final_dt[f'shift_{24}'] = final_dt.Power.shift(24).fillna(method='bfill')
# final_dt[f'shift_{48}'] = final_dt.Power.shift(48).fillna(method='bfill')
# shift_dt['power'] = final_dt['Power'].values
#
#
# correlations = shift_dt.corr()['power'] #.sort_values(ascending=False)
# corr_list = []
#
# for index in correlations.index:
#     if 0.70 < correlations[index]:
#         corr_list.append(index)
#
# shift_dt = shift_dt[corr_list]
# print(shift_dt.columns)
final_dt = final_dt.drop(['6/23/2023', '6/21/2023', '6/20/2023', '6/15/2023', '6/16/2023', '6/17/2023',
                          '6/18/2023', '6/19/2023', '5/24/2023', '2/23/2023', '6/28/2023', '6/30/2023'])
# final_dt.to_csv('datasets/sf_dataset.csv')  # Based on GMT+0 time zone
