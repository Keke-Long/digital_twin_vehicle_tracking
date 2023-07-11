import pandas as pd
import matplotlib.pyplot as plt

camera_num = '0001'
# 读取CSV文件并解析时间列
data = pd.read_csv(f'Trajectory/{camera_num}.csv', parse_dates=['time'])
data = data.dropna(subset=['id'])

# 按照id和时间排序数据
data = data.sort_values(by=['id', 'time'])

# 按照id分组
groups = data.groupby('id')

# 创建第一个图形和子图
fig1, ax1 = plt.subplots(figsize=(10, 10))
for group_name, group_data in groups:
    ax1.plot(group_data['x_utm'], group_data['y_utm'], label=f'ID: {group_name}')
    ax1.set_xlabel('x_utm (m)')
    ax1.set_ylabel('y_utm (m)')
    ax1.axis('equal')
plt.savefig(f'Trajectory/{camera_num}.png')
plt.show()




