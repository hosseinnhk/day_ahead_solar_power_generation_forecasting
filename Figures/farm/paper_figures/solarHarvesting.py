import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
import xarray as xr
import matplotlib.ticker as ticker

sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman')
sns.set(style='ticks')

# data = pd.read_csv('solarPower_test9.csv')
# data['datetime'] = pd.to_datetime(data['date'] + ' ' + data['time'])
# data.set_index('datetime', inplace=True)
# data_2022 = data[data.index.year == 2022]
# data_2022['month'] = data_2022.index.month
# monthly_power = data_2022.groupby(['month']).sum()
# monthly_power = monthly_power['Power']
# monthly_power_list = [monthly_power[month] for month in range(1, 13)]
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def data_set_creator(city, filename, expvr=True):
    dt = xr.open_dataset(filename).resample(time='1D').sum().to_dataframe()/3600
    dt = dt.reset_index(drop=False)
    dt = dt.set_index('time')
    if expvr:
        dt = dt.drop(['longitude', 'latitude', 'expver'], axis=1)
    else:
        dt = dt.drop(['longitude', 'latitude'], axis=1)
    dt.rename(columns={'ssr': f'{city}'}, inplace=True)  # re_suffix(f'{city}')
    return dt


madrid = data_set_creator('Madrid', 'surface_net_solar_radiation_2010_2023_Madrid.nc', expvr=False)
madrid = madrid.reset_index()
madrid['time'] = pd.to_datetime(madrid['time'])
madrid = madrid[madrid['time'].dt.year == 2021]
madrid['month'] = madrid['time'].dt.month
madrid = madrid.groupby(['month']).sum()
madrid = madrid['Madrid']

tallinn = data_set_creator('Tallinn', 'surface_net_solar_radiation_2010_2023_Tallinn.nc', expvr=False)
tallinn = tallinn.reset_index()
tallinn['time'] = pd.to_datetime(tallinn['time'])
tallinn = tallinn[tallinn['time'].dt.year == 2021]
tallinn['month'] = tallinn['time'].dt.month
tallinn = tallinn.groupby(['month']).sum()
tallinn = tallinn['Tallinn']

Munich = data_set_creator('Munich', 'surface_net_solar_radiation_2010_2022_Munich.nc', expvr=False)
Munich = Munich.reset_index()
Munich['time'] = pd.to_datetime(Munich['time'])
Munich = Munich[Munich['time'].dt.year == 2021]
Munich['month'] = Munich['time'].dt.month
Munich = Munich.groupby(['month']).sum()
Munich = Munich['Munich']


colors = sns.color_palette("crest", n_colors=12)
# sns.barplot(x=months, y=monthly_power_list, palette=colors)
# plt.xticks(range(0, 12), months)
# plt.ylabel('Total Solar Power (Wh)')
# plt.title('Total harvested Solar Power by Month (2022)')
# sns.despine()
# plt.savefig('Total harvested Solar Power by Month.svg')
# plt.show()

# fig, ax = plt.subplots(figsize=(10, 6))
# cmap = sns.cubehelix_palette(start=2.8, rot=0.1, dark=0.2, light=0.85, reverse=False, as_cmap=True)
# bars = ax.bar(months, monthly_power, color=cmap(monthly_power))
# ax.set_ylabel('Total Solar Power', fontsize=20)
# ax.set_xlabel('Month', fontsize=20)
# ax.set_title('Total Solar Power by Month (2022)', fontsize=20)
# sns.despine()
# ax.tick_params(axis='both', which='both', length=0)
# ax.tick_params(axis='x', labelsize=14, rotation=45)
# ax.tick_params(axis='y', labelsize=20)
# fig.tight_layout()
# plt.savefig('totalSolarPower2022.svg')
# plt.show()


angles = np.linspace(0, 2 * np.pi, len(months), endpoint=False).tolist()
angles.append(angles[0])
values = tallinn.to_list() #monthly_power.tolist()
values_m = madrid.tolist()
values_mu = Munich.tolist()
values.append(values[0])
values_m.append(values_m[0])
values_mu.append(values_mu[0])
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
ax.fill(angles, values, color='skyblue', alpha=0.5)
ax.fill(angles, values_m, color='orange', alpha=0.2)
ax.fill(angles, values_mu, color='red', alpha=0.2)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(months, fontsize=17)
ax.set_ylabel('Total Solar Irradiance (W/m2)', fontsize=16)
# ax.set_title('Total Solar Power by Month (2022)', fontsize=15)
ax.yaxis.set_label_coords(-0.1, 0.5)
# ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14, )
ax.legend(['Tallin', 'Madrid', 'Munich'], loc='upper right', fontsize=20, bbox_to_anchor=(1.2, 1.2))
# ax.text(angles[0], values[0], f'{values[0]:.2e}', fontsize=14, ha='center', va='bottom')
# for i in range(2, len(angles)-3):
#     if i > 3 and i<6:
#         ax.text(angles[i], values[i]-500000, f'{values[i]:.2e}', fontsize=14, ha='center', va='bottom')
#     else:
#         ax.text(angles[i], values[i], f'{values[i]:.2e}', fontsize=14, ha='center', va='bottom')
# ax.text(angles[i], values[i], f'{values[i]:.2e}', fontsize=14, ha='center', va='bottom')
# ax.yaxis.set_major_formatter(ScalarFormatter())
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-1, 1))
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.offsetText.set_fontsize(18)
# ax.set_xticklabels(months, rotation=45, ha='right')
plt.tight_layout()
plt.savefig('radarplotSolarpower2.svg')
plt.show()