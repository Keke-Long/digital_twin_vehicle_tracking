'''
图像坐标转世界坐标，将世界坐标信息也添加到csv
Convert image coordinates to world coordinates and add world coordinate information to csv, then visulize
'''

import numpy as np
import cv2 as cv
import utm
import pandas as pd
import re
from tqdm import tqdm
import matplotlib.pyplot as plt



Video1_GPS_info = np.float32([
[43.06709934148272, -89.38579767059579],
[43.066914588078255, -89.38606686888122],
[43.06688398855363, -89.38581754091584],
[43.06681759522982, -89.38599432495353],
[43.066790162099664, -89.38607948508509],
[43.06681324698103, -89.3856114403042],
])
Video1_imgPts = np.float32([
[685,490],
[554,676],
[1052,516],
[1114,658],
[1163,871],
[1289,472],
])



# Convert latitude and longitude to UTM
def GPS2UTM(GPS_info):
    utmPtsl = []
    for i in range(len(GPS_info)):
        lat, lon = GPS_info[i,0], GPS_info[i,1]
        utm_ = utm.from_latlon(lat, lon) # from_latlon(latitude, longitude, force_zone_number=None, force_zone_letter=None)
        utm_x = utm_[0]
        utm_y = utm_[1]
        utm_zone = utm_[2]
        utm_band = utm_[3]
        utmPtsl.append([utm_x,utm_y,0])
        # print("utm_x: %s, utm_y: %s, utm_zone: %s, utm_band: %s" % (utm_x, utm_y, utm_zone, utm_band))
        # lat, lon = utm.to_latlon(utm_x, utm_y, utm_zone, utm_band)
    utmPts = np.asarray(utmPtsl, dtype=np.float32)
    return utmPts


def cvt_pos(u , v, mat):
    x = (mat[0][0]*u+mat[0][1]*v+mat[0][2])/(mat[2][0]*u+mat[2][1]*v+mat[2][2])
    y = (mat[1][0]*u+mat[1][1]*v+mat[1][2])/(mat[2][0]*u+mat[2][1]*v+mat[2][2])
    return (round(x,2), round(y,2))



Video_num = "video1"
utmPts1 = GPS2UTM(globals()["Video1_GPS_info"])
imgPts1 = np.array([globals()["Video1_imgPts"]])
w = 1920
h = 1080

# Camera Calibration
size = (w, h)
rms1, camera_matrix1, dist1, rvec1, tvec1 = cv.calibrateCamera([utmPts1], [imgPts1], size, None, None)
new_camera_matrix1, _ = cv.getOptimalNewCameraMatrix(camera_matrix1, dist1, size, 1, size)


# Camera Position & Camera Rotation
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]) # 计算旋转角度
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])# 返回欧拉角

rvec1_ = rvec1[0]
R, _ = cv.Rodrigues(rvec1_)
camera_position_utm = -np.linalg.inv(R) @ tvec1
print(camera_position_utm)
camera_position_utm_1d = camera_position_utm.flatten()

eulerAngles = rotationMatrixToEulerAngles(R) # 从旋转矩阵获取欧拉角
eulerAngles_deg = np.degrees(eulerAngles) # 转换为度
with open(f"./camera_calibration_info/camera_parameter_{Video_num}.txt", "w") as file:
    file.write(f"Camera UTM position: \n")
    for value in camera_position_utm_1d:
        file.write(f"{value}\n")
    file.write(f"Rotation angles (Radians) for X, Y, Z: {eulerAngles} \n")
    file.write(f"Rotation angles (Degrees) for X, Y, Z: {eulerAngles_deg} \n")


# Pixel coordinate to World coordinate 第一个参数是转化前坐标，第二个参数是转化后坐标
hom1, _ = cv.findHomography(imgPts1, utmPts1, cv.RANSAC, 5)

file_dir = "./step5_Trajectory/" + Video_num + ".csv"
d = pd.read_csv(file_dir)
d = d.loc[:, ~d.columns.str.contains('^Unnamed')]
d["x_utm"] = np.nan
d["y_utm"] = np.nan
with tqdm(total=len(d)) as pbar:
    for i, row in d.iterrows():
        d.at[i, 'x_utm'], d.at[i, 'y_utm'] = cvt_pos(getattr(row, 'x_pix'), getattr(row, 'y_pix'), hom1)  # pixel to utm
d.x_utm = d.x_utm.round(3)
d.y_utm = d.y_utm.round(3)
d.to_csv(path_or_buf = file_dir, index=False)



# Visualization
def visualize_trajectories(csv_file):
    data = pd.read_csv(csv_file)
    plt.figure(figsize=(10, 8))
    for obj_class in data['class'].unique():
        obj_data = data[data['class'] == obj_class]
        grouped_data = obj_data.groupby('id')

        for name, group in grouped_data:
            group = group.sort_values(by='time')
            if obj_class == 0 or obj_class == 1:  # People or bicycles
                linestyle = 'dashed'
            else:
                linestyle = 'solid'
            plt.plot(group['x_utm'], group['y_utm'], linestyle=linestyle, label=f'ID {name} - Class {obj_class}')
    plt.xlabel('x_utm')
    plt.ylabel('y_utm')
    plt.legend()
    plt.show()

visualize_trajectories(file_dir)