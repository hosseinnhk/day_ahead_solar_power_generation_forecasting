"""
time': specifies the time format used in the data (ISO8601)
'direct_radiation': units of direct radiation data (W/m²)
'diffuse_radiation': units of diffuse radiation data (W/m²)
'direct_normal_irradiance': units of direct normal irradiance data (W/m²)
'cloudcover_low': units of low cloud cover data (%)
'cloudcover_mid': units of mid-level cloud cover data (%)
'cloudcover_high': units of high-level cloud cover data (%)'
"""

import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

'''forecast values'''
url = "https://api.open-meteo.com/v1/forecast"
start_day = "2023-07-01"
end_day = "2023-07-15"
latitude = "59.39"  # 59.10 , farm
longitude = "24.66"  # "24.30"
params = {
    "latitude": latitude,
    "longitude": longitude,
    "hourly":
    # "is_day,"
    #     "shortwave_radiation,"
    # "direct_radiation,"
    # "diffuse_radiation,"
    # "direct_normal_irradiance,"
    # "terrestrial_radiation,"
    # "uv_index,"
    # "uv_index_clear_sky,"
    # "shortwave_radiation_instant,"
    # "direct_radiation_instant,"
    # "diffuse_radiation_instant,"
    # "direct_normal_irradiance_instant,"
    # "terrestrial_radiation_instant,"
    # "temperature_2m,"
    # "dewpoint_2m,"
    # "relativehumidity_2m,"
    # "surface_pressure,"
    # "windspeed_10m,"
    # "winddirection_10m,"
    # "cloudcover,"
    # "cloudcover_low,"
    # "cloudcover_mid,"
    # "cloudcover_high,"
    # "cloudcover_400hPa,"
    # "visibility,"
    # "precipitation_probability,"
    # "precipitation,"
    # "rain,"
    # "showers,"
    # "snowfall,"
    # "snow_depth,"
    "weathercode,",
    "models": "best_match",
    # "daily": "sunrise,sunset",
    "forecast_days": "1",
    "start_date": start_day,
    "end_date": end_day,
    "timezone": "Europe/Moscow"
}

# response = requests.request("GET", url, params=querystring)
response = requests.get(url, params=params)

if response.ok:
    # data_dict = json.loads(response.text)
    data_dict = response.json()
    hourly_data = data_dict['hourly']
    # daily_data = data_dict['daily']
    df = pd.DataFrame(hourly_data)
    df['date'] = df['time'].str.split('T').str[0]
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%H:%M:%S')
    # df = df[['date', 'time', 'shortwave_radiation']]
    # df = df[['date', 'time', 'is_day', 'shortwave_radiation', 'shortwave_radiation_instant',
    #          'direct_radiation', 'direct_radiation_instant', 'diffuse_radiation',
    #          'diffuse_radiation_instant', 'direct_normal_irradiance', 'direct_normal_irradiance_instant',
    #          'terrestrial_radiation', 'terrestrial_radiation_instant', 'uv_index_clear_sky', 'uv_index',
    #          'temperature_2m', 'dewpoint_2m', 'relativehumidity_2m', 'surface_pressure', 'windspeed_10m',
    #          'winddirection_10m', 'cloudcover', 'cloudcover_400hPa', 'cloudcover_low', 'cloudcover_mid',
    #          'cloudcover_high','visibility', 'precipitation_probability', 'precipitation', 'rain', 'showers',
    #          'snowfall', 'snow_depth', 'weathercode']]
    # df.to_csv(f'f_weather_{latitude}_{longitude}.csv')
    # print(df.head)
    # print(daily_data)
else:
    print(f"Request failed with status code {response.status_code}")
''''''
print(df)


'''Historical values'''
# url = 'https://archive-api.open-meteo.com/v1/archive'
# h_start_day = '2022-11-04'
# h_end_day = '2023-06-31'
# # latitude = "59.39"   # 59.42
# # longitude = "24.66"  # "24.80"
#
# params = {
#     "latitude": latitude,
#     "longitude": longitude,
#     'start_date': h_start_day,
#     'end_date': h_end_day,
#     'hourly':
#         'shortwave_radiation,'
#     #     'direct_radiation,'
#     #     'diffuse_radiation,'
#     #     'direct_normal_irradiance,'
#     #     'temperature_2m,'
#     #     'dewpoint_2m,'
#     #     'relativehumidity_2m,'
#     #     'surface_pressure,'
#     #     'cloudcover,'
#     #     'cloudcover_low,'
#     #     'cloudcover_mid,'
#     #     'cloudcover_high,'
#     #     'windspeed_10m,'
#     #     'winddirection_10m,'
#     #     'precipitation,'
#     #     'rain,'
#     #     'snowfall,'
#     #     'weathercode'
#     ,
#     # 'daily':
#     #     'weathercode,'
#     #     'sunrise,'
#     #     'sunset,'
#     #     'shortwave_radiation_sum,'
#     #     'precipitation_sum,'
#     #     'rain_sum,'
#     #     'snowfall_sum,'
#     #     'precipitation_hours',
#     'models': 'best_match',
#     "timezone": "Europe/Moscow",
# }
#
# response = requests.get(url, params=params)
#
# if response.ok:
#     json_data = response.json()
#     dh = pd.DataFrame(json_data['hourly'])
#     # df_daily = pd.DataFrame(json_data['daily'])
#     dh['date'] = dh['time'].str.split('T').str[0]
#     dh['time'] = pd.to_datetime(dh['time']).dt.strftime('%H:%M:%S')
#     dh = dh[['date', 'time', 'shortwave_radiation', ]]
#     # 'direct_radiation', 'diffuse_radiation',
#     # 'direct_normal_irradiance', 'temperature_2m', 'dewpoint_2m',
#     # 'relativehumidity_2m', 'surface_pressure', 'windspeed_10m',
#     # 'winddirection_10m', 'cloudcover',
#     # 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high',
#     # 'precipitation', 'rain', 'snowfall', 'weathercode']]
# else:
#     print(f"Request failed with status code {response.status_code}")
# ''''''
# df.rename(columns={'shortwave_radiation': 'forecast_ghi'}, inplace=True)
# dh.rename(columns={'shortwave_radiation': 'actual_ghi'}, inplace=True)
# dfh = pd.concat([dh, df['forecast_ghi']], axis=1)
# dfh = dfh.fillna(0)
# dfh.to_csv(f'weather_{latitude}_{longitude}.csv')
# dh.to_csv(f'pvfarm_{h_start_day}_{h_end_day}.csv')
# print(df.columns)
