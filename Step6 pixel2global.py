import numpy as np
import cv2 as cv
import utm
import pandas as pd
import re
from tqdm import tqdm

#%% Video 1
# 经纬度坐标 #维度，经度
Video0001_GPS_info = np.float32([
[43.046030, -89.471974],
[43.045939, -89.471795],
[43.045898, -89.472610],
[43.045627, -89.470799],
[43.045880, -89.472123],
[43.046021, -89.472448],
])
# 像素坐标
Video0001_imgPts = np.float32([
[110, 83],
[160, 64],
[331, 212],
[217, 35],
[215, 89],
[124, 152],
])

#%% Drone 2
Video0002_GPS_info = np.float32([
[43.037166, -89.453078],
[43.037043, -89.453251],
[43.036550, -89.451875],
[43.036425, -89.452049],
])
Video0002_imgPts = np.float32([
[77, 231],
[328, 203],
[34, 41],
[88, 36],
])

#%% Drone 3
Video0003_GPS_info = np.float32([
[43.034882, -89.443154],
[43.034913, -89.442847],
[43.035118, -89.442874],
[43.035178, -89.441336],
[43.034658, -89.440904],
[43.034533, -89.443335],
])
Video0003_imgPts = np.float32([
[61, 176],
[121, 134],
[20, 130],
[161, 92],
[290, 109],
[331, 205],
])


#%% 经纬度转UTM
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
    return (round(x,2) - 300000, round(y,2) - 4770000)


Video_num = "0003"

utmPts1 = GPS2UTM(globals()["Video" + Video_num + "_GPS_info"])
imgPts1 = np.array([globals()["Video" + Video_num + "_imgPts"]])
w = 352
h = 240

# Camera Calibration
size = (w, h)
rms1, camera_matrix1, dist1, rvec1, tvec1 = cv.calibrateCamera([utmPts1], [imgPts1], size, None, None)
new_camera_matrix1, _ = cv.getOptimalNewCameraMatrix(camera_matrix1, dist1, size, 1, size)

# Pixel coordinate to World coordinate 第一个参数是转化前坐标，第二个参数是转化后坐标
hom1, _ = cv.findHomography(imgPts1, utmPts1, cv.RANSAC, 5)

file_dir = "./Trajectory/" + Video_num + ".csv"
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