'''
拟合线性回归模型：
    使用线性回归拟合 x 坐标与 y 坐标的关系，以了解它们之间的线性关系。
    计算回归线的斜率 (m) 和截距 (b)。
点到直线的投影：
    定义一个函数来计算点到给定直线的投影坐标。
    对每个车辆的轨迹点进行处理，将它们投影到具有相同斜率的各自的直线上。
确定范围并添加额外的轨迹点：
    确定所有轨迹中 x 坐标的最小值和最大值。
    对于那些没有从 x 最小值开始或没有延伸到 x 最大值的轨迹，计算并添加相应的轨迹点。
合并可能来自同一辆车的轨迹：
    分组统计每辆车的轨迹信息，包括时间的最小和最大值，以及 x 和 y 坐标的起始和结束值。
    设置时间和距离阈值，用于判断是否将两条轨迹视为同一辆车的轨迹。
    如果两条轨迹在时间和空间上都足够接近，将它们合并。
'''

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

def plot_trj(data):
    data = data.dropna(subset=['id'])
    groups = data.groupby('id')
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    for group_name, group_data in groups:
        ax1.plot(group_data['x_utm'], group_data['y_utm'], label=f'ID: {group_name}')
        ax1.set_xlabel('x_utm (m)')
        ax1.set_ylabel('y_utm (m)')
        ax1.axis('equal')
    plt.show()


def plot_speed(data):
    groups = data.groupby('id')
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for group_name, group_data in groups:
        ax2.plot(group_data['time'], group_data['speed'], label=f'ID: {group_name}')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('speed (m/s)')
    #ax2.legend(loc='upper right', ncol=2)
    plt.title('speed over Time for Each Vehicle')
    plt.show()


# Load the data
camera_num = '0001'
data = pd.read_csv(f'Trajectory/{camera_num}.csv')

# Calculate speed for each vehicle ID
delta_time = data.groupby('id')['time'].diff()
delta_x = data.groupby('id')['x_utm'].diff()
delta_y = data.groupby('id')['y_utm'].diff()
data['speed'] = (delta_x ** 2 + delta_y ** 2) ** 0.5 / delta_time

# plot_trj(data)
# plot_speed(data)

# Step 1: Fit a linear regression model to the data
reg = LinearRegression().fit(data[['x_utm']], data['y_utm'])
m = reg.coef_[0]
b = reg.intercept_

# Function to project a point onto a line
def project_point_to_line(x, y, m, b):
    x_proj = (x + m * (y - b)) / (1 + m**2)
    y_proj = m * x_proj + b
    return x_proj, y_proj

# Project each vehicle's trajectory onto its own line with the same slope
for vehicle_id in data['id'].unique():
    vehicle_data = data[data['id'] == vehicle_id]
    vehicle_b = np.mean(vehicle_data['y_utm'] - m * vehicle_data['x_utm'])
    for idx in vehicle_data.index:
        x, y = data.loc[idx, 'x_utm'], data.loc[idx, 'y_utm']
        x_proj, y_proj = project_point_to_line(x, y, m, vehicle_b)
        data.at[idx, 'x_utm'] = x_proj
        data.at[idx, 'y_utm'] = y_proj


# Step 2: Determine the range and add points to trajectories that are not within the range
x_min = data['x_utm'].min()
x_max = data['x_utm'].max()
additional_rows = []

for vehicle_id in data['id'].unique():
    vehicle_data = data[data['id'] == vehicle_id]
    if vehicle_data['x_utm'].min() > x_min:
        y_at_x_min = m * x_min + (np.mean(vehicle_data['y_utm'] - m * vehicle_data['x_utm']))
        additional_rows.append({'id': vehicle_id, 'x_utm': x_min, 'y_utm': y_at_x_min})
    if vehicle_data['x_utm'].max() < x_max:
        y_at_x_max = m * x_max + (np.mean(vehicle_data['y_utm'] - m * vehicle_data['x_utm']))
        additional_rows.append({'id': vehicle_id, 'x_utm': x_max, 'y_utm': y_at_x_max})

if additional_rows:
    data = data.append(additional_rows, ignore_index=True)


# Step 3: Merge trajectories that are likely from the same vehicle
# vehicle_info = data.groupby('id').agg({
#     'time': ['min', 'max'],
#     'x_utm': ['first', 'last'],
#     'y_utm': ['first', 'last']
# }).reset_index()
#
# vehicle_info.columns = ['id', 'start_time', 'end_time', 'start_x', 'start_y', 'end_x', 'end_y']
# time_threshold = data['time'].diff().median() * 2
# distance_threshold = np.sqrt((data['x_utm'].diff().median())**2 + (data['y_utm'].diff().median())**2) * 2
# merge_mapping = {}
#
# for idx, row in vehicle_info.iterrows():
#     potential_matches = vehicle_info[
#         (abs(vehicle_info['end_time'] - row['start_time']) < time_threshold) &
#         (np.sqrt((vehicle_info['end_x'] - row['start_x'])**2 + (vehicle_info['end_y'] - row['start_y'])**2) < distance_threshold)
#     ]
#     for _, match_row in potential_matches.iterrows():
#         merge_mapping[match_row['id']] = row['id']
# data['id'] = data['id'].replace(merge_mapping)


data['id'] = data['id'].astype(int)
data.to_csv(f"Trajectory/{camera_num}_after step 7.csv", index=False)

plot_trj(data)
plot_speed(data)

