import os
from os import path as osp
import cv2
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from ultralytics import YOLO

from utils import *

parser = argparse.ArgumentParser(description='Parameter settings to run the file')
parser.add_argument('--data_root_dir', type=str, default='./data', help='Root path to the data files')
parser.add_argument('--result_root_dir', type=str, default='./results', help='Root path to the result files')
parser.add_argument('--is_extract_frames', action="store_true", 
                    default=False, help='True when extracting frames from video, False when not extracting frames from video')

args = parser.parse_args()


class ProcessVideo:
    def __init__(self, args, video_id):
        self.out_root_dir = args.result_root_dir
        self.data_root_dir = args.data_root_dir
        self.video_id = video_id

    def track_object(self):
        # input_file = osp.join(self.data_root_dir, 'videos', self.video_id + '.mp4')
        input_file = osp.join(self.data_root_dir, 'videos', self.video_id + '.asf')
        model = YOLO('yolov8n.pt')
        project_path = osp.join(self.out_root_dir, 'yolo_results', self.video_id)
        out_name = 'track_results'
        model.track(source=input_file,
                    tracker='bytetrack.yaml',
                    classes=[0,1,2],
                    project=project_path,
                    name=out_name,
                    save=True,
                    conf=0.3,
                    iou=0.5,
                    show=True,
                    save_txt=True)
    
    def trackresult_to_trj(self, frames=15):
        camera_list = [self.video_id]
        track_result = osp.join(self.out_root_dir, 'yolo_results', self.video_id, 'track_results')

        for camera in camera_list: #,'0002','0003'
            print(camera)
            path = os.path.join(track_result, 'labels')
            print("path", path)

            data = pd.DataFrame()

            frame_count = len([f for f in os.listdir(path) if osp.isfile(osp.join(path, f))])

            file_list = os.listdir(path)
            for file in tqdm(file_list):
                #print('file', file)

                d = pd.read_table(osp.join(path, file), sep='\s', names=['class', 'x_pix', 'y_pix', 'w_pix', 'h_pix', 'id'])

                if d['id'].isnull().any() or d['class'].isnull().any() or d['x_pix'].isnull().any() or d['y_pix'].isnull().any():
                    continue

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

            out_dir = osp.join(self.out_root_dir, 'trajectory', camera + ".csv")
            data.to_csv(path_or_buf=out_dir, columns=['id', 'time', 'frame_num', 'x_pix', 'y_pix', 'w_pix', 'h_pix'], index=False)


    def camera_calibration(self):

        video_gps, video_pxl = pixel_to_global()

        utmPts1 = GPS2UTM(video_gps)
        imgPts1 = np.array([video_pxl])
        w = 352
        h = 240

        # Camera Calibration
        size = (w, h)
        # TODO: cv2.error: OpenCV(4.10.0) /io/opencv/modules/calib3d/src/calibration.cpp:1783: error: (-5:Bad argument) There should be less vars to optimize (having 15) than the number of residuals (12 = 2 per point) in function 'cvCalibrateCamera2Internal'
        rms1, camera_matrix1, dist1, rvec1, tvec1 = cv2.calibrateCamera([utmPts1], [imgPts1], size, None, None)
        new_camera_matrix1, _ = cv2.getOptimalNewCameraMatrix(camera_matrix1, dist1, size, 1, size)

        rvec1_ = rvec1[0]
        R, _ = cv2.Rodrigues(rvec1_)
        camera_position_utm = -np.linalg.inv(R) @ tvec1
        print(camera_position_utm)
        camera_position_utm_1d = camera_position_utm.flatten()

        eulerAngles = rotationMatrixToEulerAngles(R) # Return the Euler angles
        eulerAngles_deg = np.degrees(eulerAngles) # Return the Euler angles in degrees
        with open('camera_parameter_{}.txt'.format(self.video_id), "w") as file:
            file.write(f"Camera UTM position: \n")
            for value in camera_position_utm_1d:
                file.write(f"{value}\n")
            file.write(f"Rotation angles (Radians) for X, Y, Z: {eulerAngles} \n")
            file.write(f"Rotation angles (Degrees) for X, Y, Z: {eulerAngles_deg} \n")

        # Pixel coordinate to World coordinate (UTM)
        hom1, _ = cv2.findHomography(imgPts1, utmPts1, cv2.RANSAC, 5)

        file_dir = osp.join(self.out_root_dir, "trajectory")
        d = pd.read_csv(osp.join(file_dir, self.video_id + ".csv"))
        d = d.loc[:, ~d.columns.str.contains('^Unnamed')]
        d["x_utm"] = np.nan
        d["y_utm"] = np.nan
        with tqdm(total=len(d)) as pbar:
            for i, row in d.iterrows():
                d.at[i, 'x_utm'], d.at[i, 'y_utm'] = cvt_pos(getattr(row, 'x_pix'), getattr(row, 'y_pix'), hom1)  # pixel to utm
        d.x_utm = d.x_utm.round(3)
        d.y_utm = d.y_utm.round(3)
        d.to_csv(path_or_buf=osp.join(file_dir, self.video_id + "_calibrated.csv"), index=False)


    def process_traj(self):

        # Load the data
        data = pd.read_csv(osp.join(self.out_root_dir, 'trajectory', self.video_id + '_calibrated.csv'))
        # data = pd.read_csv(osp.join(self.out_root_dir, 'trajectory', self.video_id + '.csv'))

        # Calculate speed for each vehicle ID
        data = data.sort_values(by='frame_num')
        delta_time = data.groupby('id')['time'].diff()
        delta_x = data.groupby('id')['x_utm'].diff()
        delta_y = data.groupby('id')['y_utm'].diff()
        data['speed'] = (delta_x ** 2 + delta_y ** 2) ** 0.5 / delta_time

        # Step 1: Fit a linear regression model to the data
        reg = LinearRegression().fit(data[['x_utm']], data['y_utm'])
        m = reg.coef_[0]
        b = reg.intercept_

        # Project each vehicle's trajectory onto its own line with the same slope
        for vehicle_id in data['id'].unique():
            vehicle_data = data[data['id'] == vehicle_id]
            vehicle_b = np.mean(vehicle_data['y_utm'] - m * vehicle_data['x_utm'])
            for idx in vehicle_data.index:
                x, y = data.loc[idx, 'x_utm'], data.loc[idx, 'y_utm']
                x_proj, y_proj = project_point_to_line(x, y, m, vehicle_b)
                data.at[idx, 'x_utm'] = x_proj
                data.at[idx, 'y_utm'] = y_proj

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
        vehicle_info = data.groupby('id').agg({
            'time': ['min', 'max'],
            'x_utm': ['first', 'last'],
            'y_utm': ['first', 'last']
        }).reset_index()
        
        vehicle_info.columns = ['id', 'start_time', 'end_time', 'start_x', 'start_y', 'end_x', 'end_y']
        time_threshold = data['time'].diff().median() * 2
        distance_threshold = np.sqrt((data['x_utm'].diff().median())**2 + (data['y_utm'].diff().median())**2) * 2
        merge_mapping = {}
        
        for idx, row in vehicle_info.iterrows():
            potential_matches = vehicle_info[
                (abs(vehicle_info['end_time'] - row['start_time']) < time_threshold) &
                (np.sqrt((vehicle_info['end_x'] - row['start_x'])**2 + (vehicle_info['end_y'] - row['start_y'])**2) < distance_threshold)
            ]
            for _, match_row in potential_matches.iterrows():
                merge_mapping[match_row['id']] = row['id']
        data['id'] = data['id'].replace(merge_mapping)


        data['id'] = data['id'].astype(int)
        data.to_csv(osp.join(self.out_root_dir, 'trajectory', self.video_id + '_processed.csv'), index=False)

        def plot_traj(data):
            plt.figure(figsize=(10, 10))
            for id in data.id.unique():
                traj = data[data.id == id]
                plt.plot(traj.x_utm, traj.y_utm, label=f"ID: {id}")
            plt.xlabel("X (UTM)")
            plt.ylabel("Y (UTM)")
            plt.legend()
            plt.savefig(osp.join(self.out_root_dir, 'trajectory', self.video_id + '_traj.jpg'))
            # plt.show() 


        def plot_speed(data):
            groups = data.groupby('id')
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            for group_name, group_data in groups:
                ax2.plot(group_data['time'], group_data['speed'], label=f'ID: {group_name}')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('speed (m/s)')
            plt.title('speed over Time for Each Vehicle')
            plt.savefig(osp.join(self.out_root_dir, 'trajectory', self.video_id + '_speed.jpg'))
            # plt.show()

        plot_traj(data)
        plot_speed(data)

    def label_to_pic(self):
        img_folder = osp.join(self.out_root_dir, 'frames', self.video_id)
        label_folder = osp.join(self.out_root_dir, 'yolo_results', self.video_id, 'track_results', 'labels')
        out_folder = osp.join(self.out_root_dir, 'labeled_results', self.video_id)
        speed_lookup = load_speed_data(osp.join(self.out_root_dir, 'trajectory', '{}_processed.csv'.format(self.video_id)))

        for label_num in range(1501):
            label_path = osp.join(label_folder, "{}_{}.txt".format(self.video_id, label_num))
            image_path = osp.join(img_folder, "{}.jpg".format(label_num))
            print('image', "{}.jpg".format(label_num), 'label', "{}.txt".format(label_num))

            # Check if the label file exists
            if not osp.exists(label_path):
                print(f"Label file {label_path} does not exist!")
                continue

            timestamp = round(label_num / 15, 2)

            # Load the label file
            # img = cv2.imread(osp.join(image_path, '{}.jpg'.format(label_num)))
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                # Load labels
                with open(label_path, 'r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                # Draw a bounding box
                for x in lb:
                    # Get the coordinates of the rectangle
                    _img = xywh2xyxy(x, w, h, img, round(timestamp, 2), speed_lookup)
                
                cv2.imwrite(osp.join(out_folder, '{}.png'.format(label_num)), _img)

    def pic_to_video(self):
        folder_path = osp.join(self.out_root_dir, 'labeled_results', self.video_id)
        video_path = osp.join(self.out_root_dir, '{}.mp4'.format(self.video_id))
        fps = 15

        images = [img for img in os.listdir(folder_path) if img.endswith('.png')]
        images.sort(key=natural_keys)

        frame = cv2.imread(osp.join(folder_path, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for image in images:
            video.write(cv2.imread(osp.join(folder_path, image)))

        video.release()


def main():
    video_path = osp.join(args.data_root_dir, 'videos')
    out_path = args.result_root_dir
    video_id = "verona0001"

    offset = 10
    # start_time = 1685917196
    start_time = 500
    end_time = start_time + offset

    frames = 15
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    process_video = ProcessVideo(args, video_id)

    flg_dict = {'is_yolo': False, 'is_merge_images': False, 'is_traj': False, 'is_calibration': False, 
                'is_process_traj': True, 'is_label_to_pic': True, 'is_pic_to_video': True}

    # Extract frames from video and save them as images
    if args.is_extract_frames:
        print("Extracting frames from the video...")
        extract_frames_from_video(video_path, out_path, video_id)

    # Load the frames and perform object detection and save it to the output folder
    if flg_dict['is_yolo']:
        print("Performing object detection on the frames...")
        process_video.track_object()

    # cameras = [folder for folder in os.listdir(track_result) if osp.isdir(osp.join(out_path, track_result, folder))]

    # Merge the detected frames into a video
    if flg_dict['is_merge_images']:
        print("Merge the detected frames into a video...")
        merge_images_to_video(out_path, video_id, start_time, end_time)

    # Extract trajectory from image
    if flg_dict['is_traj']:
        print("Extracting trajectory from the image...")
        process_video.trackresult_to_trj(frames)

    # Camera calibration 
    if flg_dict['is_calibration']:
        print("Camera calibration...")
        process_video.camera_calibration()

    # Plot trajectory
    if flg_dict['is_process_traj']:
        print("Processing trajectory...")
        process_video.process_traj()
    
    # Label to picture
    if flg_dict['is_label_to_pic']:
        print("Label to picture...")
        process_video.label_to_pic()

    # Picture to video
    if flg_dict['is_pic_to_video']:
        print("Picture to video...")
        process_video.pic_to_video()


if __name__ == '__main__':
    main()