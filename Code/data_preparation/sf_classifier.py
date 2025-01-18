import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=12)
pd.set_option('display.max_columns', None)

df = pd.read_csv('datasets/sf_dataset.csv', parse_dates=['date'], index_col=['date', 'time'])
df = df[df.Power > 0]
df['weathercode'] = df['weathercode'].astype('category')  # uncomment it you want to plot weather code figures

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

grouped_data = df.groupby('weathercode')['Power']
separated_datasets = {}

# for weather_code, group in grouped_data:
#     separated_datasets[weather_code] = group.copy()
#     separated_datasets[weather_code].sample(frac=1).to_csv(f'datasets/sf_wc_{weather_code}.csv')
#
# pd.DataFrame(pd.concat((separated_datasets[51], separated_datasets[53],
#                         separated_datasets[55], separated_datasets[61],
#                         separated_datasets[63], separated_datasets[71],
#                         separated_datasets[73], separated_datasets[75]), axis=0)).sample(frac=1).to_csv('datasets/sf_wc_others.csv')

# pd.DataFrame(pd.concat((separated_datasets[61], separated_datasets[63]), axis=0)).to_csv('datasets/sf_wc_2516163.csv')
# pd.DataFrame(pd.concat((separated_datasets[3], separated_datasets[53],
#                         separated_datasets[55]), axis=0)).to_csv('datasets/sf_wc_35355.csv')

mean_power = grouped_data.mean()

std_power = grouped_data.std()
max_power = grouped_data.max()
min_power = grouped_data.min()
group_counts = grouped_data.count()

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4))

mean_power.plot(kind='bar', alpha=0.5, edgecolor='none', ax=ax1)
ax1.set_xlabel('Weather code', fontsize=16)
ax1.set_ylabel('Power (W)', fontsize=16)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
ax1.grid(True)

ax2 = ax1.twinx()
ax2.scatter(mean_power.index.codes, max_power, c='orange', marker='o', label='Max Power', alpha=0.7)
ax2.scatter(mean_power.index.codes, min_power, c='purple', marker='o', label='Min Power', alpha=0.5)
ax2.set_ylabel('Power (Min-Max)', fontsize=16)

ax2.tick_params(axis='y', labelsize=14)
ax2.grid(False)
group_counts.plot(kind='bar', color='green', alpha=0.5, edgecolor='none', ax=ax3)
ax3.set_ylabel('Count', fontsize=16)
ax3.set_xlabel('Weather code', fontsize=16)
ax3.tick_params(axis='x', labelsize=14)
ax3.tick_params(axis='y', labelsize=14)
fig.tight_layout()
# plt.savefig('weathercode.svg')
plt.show()


