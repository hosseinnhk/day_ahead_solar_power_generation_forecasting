import pandas
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline

sns.set_style("whitegrid")
plt.rc('font', family='Times New Roman', size=18)


df = pd.read_csv('temperature.csv', parse_dates=['time'], index_col=['time'])
medians = []
for i in range(df.shape[0]):
    medians.append(np.mean(df.iloc[i, :]))
df['medians'] = medians
plt.figure(figsize=(8, 4))
for i in range(df.shape[1]):
    plt.plot(df.iloc[:, i], 'k', alpha=0.15)
plt.plot(df.medians, 'red', alpha=0.9, linewidth=1.5)
plt.ylabel('Temperature ($^\circ$C)')
# plt.ylabel('Solar irradiance (W/m$^2$)')
plt.grid(color='gray', alpha=0.4, linestyle='--', linewidth=0.6)
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.xticks(rotation=45)
# plt.xlim(pd.Timestamp('2023-08-01 00:00:00'), pd.Timestamp('2023-08-01 23:00:00'))
plt.xlim(pd.Timestamp('2023-07-31 00:00:00'), pd.Timestamp('2023-07-31 23:00:00'))
sns.set(style='ticks')
sns.despine()
plt.tight_layout()
plt.savefig('TEMP_T.svg', transparent=True)
plt.show()

mean_sun_hours_data = [
    [148, 30, 62, 56, 75, 121, 63, 63],
    [157, 59, 78, 83, 95, 133, 79, 88],
    [214, 148, 115, 152, 145, 167, 129, 126],
    [231, 217, 169, 226, 185, 201, 166, 183],
    [272, 306, 199, 271, 215, 264, 194, 222],
    [310, 294, 204, 260, 225, 285, 202, 206],
    [359, 312, 212, 246, 245, 332, 212, 217],
    [335, 256, 205, 231, 235, 298, 212, 197],
    [261, 162, 149, 175, 170, 237, 168, 139],
    [198, 88, 117, 107, 125, 195, 118, 109],
    [157, 29, 73, 60, 70, 129, 68, 62],
    [124, 21, 52, 45, 65, 112, 51, 51]
]

months = ['Jan.', 'Feb.', 'Mar.', 'Apr.', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.']
cities = ['Madrid', 'Tallinn', 'London', 'Copenhagen', 'Munich', 'Rome', 'Paris', 'Amsterdam']

# Create the DataFrame
num_points = 200
dt_i = pd.DataFrame(mean_sun_hours_data, columns=cities, index=months)
x = np.linspace(0, len(months) - 1, len(months))
x_new = np.linspace(0, len(months) - 1, num_points)
dt = pd.DataFrame({city: make_interp_spline(x, dt_i[city])(x_new) for city in dt_i.columns})

# num_other_cities = len(cities[:4]) - 1
color = ['mediumblue', 'lime', 'darkred', 'teal', 'sandybrown']
# color = sns.color_palette('pastel', 5)
marker = ['^', 'o', 's', '*', "p", '8', "d", 'D']
# Plot the smoothed data for each city
plt.figure(figsize=(12, 6))
for i, city in enumerate(cities[:5]):
    if city == 'Tallinn':
        # plt.fill_between(np.linspace(0, num_points, num_points), dt[city], color='yellow', alpha=0.2)
        plt.plot(dt[city], color='darkorange', alpha=0.8, linewidth=2)
        plt.scatter(np.linspace(0, num_points, 12), dt_i[city], s=100, label=city, color='darkorange')
    else:
        print(i)
        plt.plot(dt[city], alpha=0.8, linewidth=2, color=color[i]) #label=city,
        plt.scatter(np.linspace(0, num_points, 12), dt_i[city], s=100, label=city,
                    marker=marker[i], color=color[i], alpha=0.8)  #

# plt.xlabel('Months')
plt.ylabel('Total Sun Hours', fontsize=20)
plt.legend()
plt.grid(color='gray', alpha=0.4, linestyle='--', linewidth=0.6)
# plt.xticks(rotation=45)
plt.xlim(0, num_points)
plt.ylim(0, 380)
plt.xticks(np.linspace(0, num_points, 12), months, fontsize=18)
plt.tight_layout()
sns.despine()
plt.savefig('sun_hours3.svg', transparent=True)
plt.show()

