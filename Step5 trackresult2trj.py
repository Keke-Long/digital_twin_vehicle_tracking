#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import re
import pandas as pd
import datetime
from tqdm import tqdm

frames = 15

track_result = "./Yolo_result/"  # 文件夹的路径
cameras = [folder for folder in os.listdir(track_result) if os.path.isdir(os.path.join(track_result, folder))]

# for camera in sorted(cameras):
for camera in ['0003']: #,'0002','0003'
    path = os.path.join(track_result, camera, camera, "labels/")
    print("path", path)

    data = pd.DataFrame()

    frame_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])

    file_list = os.listdir(path)
    for file in tqdm(file_list):
        #print('file', file)

        d = pd.read_table(path + file, sep='\s', names=['class', 'x_pix', 'y_pix', 'w_pix', 'h_pix', 'id'])

        segments = re.split(r"[._]", file)
        frame_num = int(segments[1])
        frame_time = frame_num * (1 / frames)
        d.insert(3, 'frame_num', frame_num)
        d.insert(2, 'time', frame_time)
        data = pd.concat([data, d])

    w = 352
    h = 240
    data.x_pix *= w
    data.y_pix *= h
    data.w_pix *= w
    data.h_pix *= h

    data.x_pix = data.x_pix.round(3)
    data.y_pix = data.y_pix.round(3)
    data.w_pix = data.w_pix.round(3)
    data.h_pix = data.h_pix.round(3)
    data.time = data.time.round(2)

    output_dir = "./Trajectory/" + camera + ".csv"
    data.to_csv(path_or_buf=output_dir, columns=['id', 'time', 'frame_num', 'x_pix', 'y_pix', 'w_pix', 'h_pix'], index=False)