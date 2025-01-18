import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns

sns.set_style("whitegrid")
# sns.set()
plt.rc('font', family='Times New Roman', size=20)


if not os.path.exists("PVlib_plot"):
    os.mkdir("PVlib_plot")

file_path = 'D:/pythonProjects/solarPowerForecasting/mainProject/data_prepration/farm/datasets/sf_dataset.csv'
df = pd.read_csv(file_path)
df.set_index(['date', 'time'], inplace=True)
df.index = pd.to_datetime(df.index.map(lambda x: ' '.join(x)), format='%Y-%m-%d %H:%M:%S')
# print(dataset.head(5))
# print(dataset.loc['2023-04-25':'2023-04-27', 'Power'].values)
# df['date'] = pd.to_datetime(df['date'])
# df.set_index('Date', inplace=True)
# df.drop(['Current', 'Voltage'], axis=1, inplace=True)
# resampled_dataset = df.resample('5T').mean()
# resampled_dataset = resampled_dataset.fillna(0)
# resampled_dataset[resampled_dataset.Power < 0] = 0
# shifted_dataset = resampled_dataset.shift(freq='1H')
# shifted_dataset = shifted_dataset['2023-06-01':]
# print(shifted_dataset.head())
# print(shifted_dataset.tail())
# x = shifted_dataset['2023-05-02':'2023-05-04'].Power
# plt.plot(x)
# plt.xticks(rotation=45, ha='right')
# plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
# start_time = mdates.datestr2num('2023-05-02 00:00')
# end_time = mdates.datestr2num('2023-05-04 23:59')
# plt.xlim(start_time, end_time)
# plt.show()

for i, (start, end) in enumerate(zip(df.index[0::120], df.index[120::120])):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.loc[start:end, 'Power'], color='#143841')
    # print(dataset.loc[start:end, 'Power'].values)
    ax.plot(df.loc[start:end, 'pvsim'], '--', color='#E64E04')
    # ax.set_title(f'PV power output, date: {start.strftime("%Y-%m-%d")} to {end.strftime("%Y-%m-%d")}',fontsize= 20)
    # ax.set_xlabel('Time',fontsize= 20)
    ax.set_ylabel('Power (w)', fontsize=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.set_xticks(ax.get_xticks() + 0.5)
    plt.legend(['Actual power', 'Simulated power'], fontsize= 20)
    ax.set_xlim(start, end)
    # ax.grid(False)
    sns. despine()
    filename = f'PVlib_plot/nZEB_PV_{start.strftime("%Y%m%d")}_{end.strftime("%Y%m%d")}.png'
    plt.savefig(filename, format='png', dpi=500)
    # plt.show()
    # if i==5:
    #     break
    plt.close()






