import os
from os import path as osp
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utm
import re

def extract_frames_from_video(video_path, output_dir, video_id):
    # video_name = video_id + '.mp4'
    video_name = video_id + '.asf'
    cap = cv2.VideoCapture(osp.join(video_path, video_name))

    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 


    if fps < 1 or fps > 100:
        fps = 15

    video_end_time = 1685917912
    frame_interval = 1

    print(f"Video file: {video_path}, fps: {fps}, total frames: {total_frames}")

    frame_index = 0
    ret, frame = cap.read()
    frame_timestamp = 0

    while ret:
        frame_timestamp += 1
        image_path = osp.join(output_dir, 'frames', video_id, f"{int(frame_timestamp)}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Saved frame {image_path}")

        ret, frame = cap.read()
        frame_index += 1
        if frame_index > 5000:
            break
    
    cap.release()


def merge_images_to_video(out_root, camera_number, start_time, end_time):
    frame_path= osp.join(out_root, 'frames', camera_number)  # Path to the camera folder
    out_path = osp.join(out_root, 'yolo_results', camera_number, 'out_{}_{}_{}.mp4'.format(camera_number, start_time, end_time))  # Output video path

    image_files = [file for file in os.listdir(frame_path) if file.endswith('.jpg')]  # Get files ending with .jpg

    # Filter image files within the specified start and end time range
    selected_files = []
    for file in image_files:
        file_name, ext = osp.splitext(file)
        file_time = float(file_name)  # Get the standard time in seconds

        if start_time <= file_time <= end_time:
            selected_files.append(file)

    # Sort selected image files by filename to ensure chronological order in the video
    selected_files.sort(key=lambda x: float(osp.splitext(x)[0]))

    # Get width and height information from the first image
    first_image_path = osp.join(frame_path, selected_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Calculate the frame rate and corresponding start and end frames
    frame_rate = 5  # Generate 5 frames per second
    start_frame = math.floor(start_time * frame_rate)
    end_frame = math.ceil(end_time * frame_rate)

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, frame_rate, (width, height))

    # Generate the video frame by frame
    for i in range(start_frame, end_frame + 1):
        time = i / frame_rate  # Current time corresponding to the frame
        nearest_time_file = min(selected_files, key=lambda x: abs(float(os.path.splitext(x)[0]) - time))
        image_path = osp.join(frame_path, nearest_time_file)
        image = cv2.imread(image_path)
        out.write(image)

    # Release resources
    out.release()
    print("Video merge completed!")


def pixel_to_global():


    # (lat, lng)
    Video0001_GPS_info = np.float32([
        [43.046030, -89.471974],
        [43.045939, -89.471795],
        [43.045898, -89.472610],
        [43.045627, -89.470799],
        [43.045880, -89.472123],
        [43.046021, -89.472448],
    ])
    # pixel coordinates
    Video0001_imgPts = np.float32([
        [110, 83],
        [160, 64],
        [331, 212],
        [217, 35],
        [215, 89],
        [124, 152],
    ])

    # Drone 2
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

    # Drone 3
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
    return Video0001_GPS_info, Video0001_imgPts #, Video0002_GPS_info, Video0002_imgPts, Video0003_GPS_info, Video0003_imgPts


# GPS to UTM (Universal Transverse Mercator)
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


# Camera Position & Camera Rotation
def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]) # Calculate the angle of rotation
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z]) # Return the rotation angles in radians


def project_point_to_line(x, y, m, b):
    x_proj = (x + m * (y - b)) / (1 + m**2)
    y_proj = m * x_proj + b
    return x_proj, y_proj


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def extract_float(filename):
    base = os.path.basename(filename)  # Get the file name and remove the path
    float_number = os.path.splitext(base)[0]  # Remove the extension
    try:
        return float(float_number)
    except ValueError:
        raise ValueError(f"Filename {filename} does not contain a valid float number.")
    
def load_speed_data(csv_path):
    speed_data = pd.read_csv(csv_path)
    speed_lookup = speed_data.set_index(['id', 'time'])
    print('max_speed', max(speed_data['speed']))
    return speed_lookup

def get_timestamp_from_filename(filename):
    # Assuming the filename is just an integer followed by '.jpg'
    base_name = os.path.basename(filename)  # Get the base name, e.g., '1.jpg'
    timestamp = int(os.path.splitext(base_name)[0])  # Split the extension and convert to an integer
    return timestamp

def speed_to_color(speed, max_speed):
    normalized_speed = speed / max_speed
    color_hue = (1.0 - normalized_speed) * 120  # H value ranges from red (speed = 0) to green (speed = max_speed)
    color = cv2.cvtColor(np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color)

def xywh2xyxy(x, w1, h1, img, timestamp, speed_lookup):
    if len(x) == 5:
        label, x, y, w, h = x
        id = 0
    elif len(x) == 6:  # If there is an id, unpack normally
        label, x, y, w, h, id = x
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1

    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    speed = 30
    try:
        # Ensure the accuracy of timestamp matches the frame_num
        rounded_timestamp = round(timestamp, 2)
        rounded_id = int(id)
        speed = speed_lookup.loc[(rounded_id, rounded_timestamp), 'speed']
        if isinstance(speed, pd.Series):
            speed = speed.iloc[0]

    except KeyError:
        search_times = np.arange(rounded_timestamp - 1, rounded_timestamp + 4, 0.01)
        for search_time in search_times:
            try:
                speed = speed_lookup.loc[(rounded_id, search_time), 'speed']
                if isinstance(speed, pd.Series):
                    speed = speed.iloc[0]
                break
            except KeyError:
                continue

    if speed is None or pd.isna(speed) or speed == 0:
        speed = 30

    color = speed_to_color(speed, 40) if speed is not None else (255, 255, 255)

    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color, 2)

    text_position = (int(x * w1 - w * w1 / 2), int(y * h1 - h * h1 / 2) - 10)
    cv2.putText(img, f"{speed:.2f}m/s", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img
