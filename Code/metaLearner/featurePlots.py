import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='Times New Roman', size=16)
sns.set_style("whitegrid")
from sklearn.preprocessing import MinMaxScaler

# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#
# # Create a 3D axis
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Define the base points
# base_points = np.array([
#     [1.414, 1.414, 1],
#     [-1.414, 1.414, 1],
#     [-1.414, -1.414, 1],
#     [1.414, -1.414, 1],
# ])
#
# bottom_point = np.array([0, 0, 0])
#
# ax.plot(base_points[:, 0], base_points[:, 1], base_points[:, 2], color='gray')
# ax.plot(np.append(base_points[:, 0],
#                   base_points[0, 0]), np.append(base_points[:, 1], base_points[0, 1]), np.append(base_points[:, 2],
#                                                                                                  base_points[0, 2]), color='gray')
#
# for point in base_points:
#     ax.plot([point[0], bottom_point[0]], [point[1], bottom_point[1]], [point[2], bottom_point[2]], color='gray')

metrics = np.array([[0.053, 16.95, 0.085, 0.889],
                    [0.073, 22.2, 0.112, 0.839],
                    [0.068, 18.0, 0.103, 0.865],
                    [0.060, 17.07, 0.098, 0.873],
                    [0.081, 24.51, 0.121, 0.816],
                    [0.070,	19.90, 0.106, 0.856],
                    [0.062, 17.55, 0.097, 0.876],
                    [0.119, 44.41, 0.201, 0.433]
                    ])

labels = ['Proposed',
          'Recursive-LSTM',
          'LightGBM*',
          'Classified_LightGBM',
          'LSTM*',
          'XGBoost*',
          'Average',
          'Persistence',
          ]

# def interpolate_point(base_point, tip_point, fraction):
#     return tip_point + fraction * (base_point - tip_point)
#
#
# for metrics in metrics:
#     points = [interpolate_point(base_points[i], bottom_point, metrics[i]) for i in range(4)]
#     points = [list(arr) for arr in points]
#     x = [point[0] for point in points]
#     y = [point[1] for point in points]
#     z = [point[2] for point in points]
#     surf = ax.plot_trisurf(x, y, z, antialiased=True, alpha=0.5)
#
# ax.view_init(elev=15, azim=-25)
# ax.set_xlim(-1.5, 1.5)
# ax.set_ylim(-1.5, 1.5)
# ax.set_zlim(0, 1)
# ax.axis('off')
# plt.show()

# sns.set_style("whitegrid")
#
# num_methods = metrics.shape[0]
# num_metrics = metrics.shape[1]
# x = np.arange(num_methods)  # label locations
# width = 0.2  # width of the bars
#
# fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#
# colors = ['darkred', 'steelblue', 'darkorange', 'green']
# metric_names = ['nMAE',	'MAPE %',	'nRMSE',	'R2 score']
#
# axes[0,0].bar(labels, metrics[:, 0], width=0.5, color='k')
# axes[0,0].set_title(metric_names[0])
# axes[0,1].bar(labels, metrics[:, 1], width=0.5, color='k')
# axes[0,1].set_title(metric_names[1])
# axes[1,0].bar(labels, metrics[:, 2], width=0.5, color='k')
# axes[1,0].set_title(metric_names[2])
# axes[1,1].bar(labels, metrics[:, 3], width=0.5, color='k')
# axes[1,1].set_title(metric_names[3])
# fig.tight_layout()
# plt.show()

num_methods = metrics.shape[0]
num_metrics = metrics.shape[1]
x = np.arange(num_methods)  # label locations

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

colors = ['darkred', 'steelblue', 'darkorange', 'green']
metric_names = ['nMAE', 'MAPE %', 'nRMSE', 'R2 score']
bar_width = 0.3

bar_positions_1 = [i for i in range(len(labels))]
bar_positions_2 = [i + bar_width for i in bar_positions_1]

for i, ax in enumerate(axes.flat):

    ax.bar(bar_positions_1, metrics[:, 2*i], width=bar_width, color='steelblue', alpha=0.7)
    ax.axhline(y=metrics[0, 2*i], color='steelblue', linestyle='--')
    ax.set_ylabel(metric_names[2*i])
    ax2 = ax.twinx()
    ax2.bar(bar_positions_2, metrics[:, 2*i + 1], width=bar_width, color='darkorange', alpha=0.55)
    ax2.axhline(y=metrics[0, 2*i + 1], color='darkorange', linestyle='--')
    ax2.set_ylabel(metric_names[2*i + 1])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.yaxis.label.set_color('darkorange')
    ax2.tick_params(axis='y', colors='darkorange')
    ax2.spines['right'].set_color('darkorange')
    # Set x-ticks in the middle of two bars and rotate them if necessary
    ax.set_xticks([i + bar_width / 2 for i in bar_positions_1])
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(False)
    ax2.grid(False)

fig.tight_layout()
# plt.savefig('metrics.svg')
plt.show()

# Add some text for labels, title and custom x-axis tick labels
# ax.set_xlabel('Methods')
# ax.set_ylabel('Values')
# ax.set_title('Metrics by different methods')
# ax.set_xticks(x + width*(num_metrics-1)/2)
# ax.set_xticklabels(labels)
# ax.legend()
# # Compute the angle for each axis
# angles = np.linspace(0, 2 * np.pi, metrics.shape[1], endpoint=False).tolist()
# angles += angles[:1]  # close the plot
#
# # Plot data
# for metric, label in zip(metrics, labels):
#     values = metric.tolist()
#     values += values[:1]  # close the plot
#     ax.plot(angles, values, label=label)
#     ax.fill(angles, values, alpha=0.25)
#
# # ax.set_thetagrids(np.degrees(angles), ["Metric 1", "Metric 2", "Metric 3", "Metric 4"])
#
# # Display the legend
# ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.05))
#
# plt.title("Radar Chart with Seaborn Style")
# plt.show()

"""feature importances"""
columns = ['solar_azimuth', 'solar_zenith', 'shortwave_radiation',
       'direct_radiation', 'direct_normal_irradiance', 'temperature_2m',
       'surface_pressure', 'windspeed_10m', 'winddirection_10m', 'cloudcover',
       'pvsim', 'Lag_24', 'Lag_25', 'Lag_48', 'Lag_72']

values = [287, 238, 89, 96, 170, 292, 306, 280, 287, 105, 180, 205, 137, 183, 200]

sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=False)
sorted_indices1 = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
sorted_values = [values[i] for i in sorted_indices]
sorted_columns = [columns[i] for i in sorted_indices1]

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_columns, sorted_values,  color='#155a79', height=0.35, alpha=0.9) #width=bar_width,
for bar in bars:
    xval = bar.get_width()
    if not np.isnan(xval):  # Check if the value is not NaN
        plt.text(xval + 1, bar.get_y() + bar.get_height()/2, round(xval, 2), va='center', ha='left', size=14)

# plt.xlabel('Columns')
# plt.ylabel('Values')
# plt.title('Bar plot of columns in descending order')
plt.xlim(0, 350)
plt.xticks(ha='right')  # Rotate labels for better visibility
plt.tight_layout()
plt.savefig('feature_importance2.svg')
plt.show()