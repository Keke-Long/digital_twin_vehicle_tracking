''' 遍历视频，yolov8保存视频及轨迹'''
from ultralytics import YOLO
import os

video_dir = "./Videos/"
cameras = [folder for folder in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, folder))]
for camera in sorted(cameras):
    camera_dir = os.path.join(video_dir, camera) + "/mp4/"
    for file in os.listdir(camera_dir):
        if file.endswith(".mp4"):
            model = YOLO('yolov8n.pt')  # load an official detection model
            input_file = os.path.join(camera_dir, file)
            project_path = "/home/ubuntu/Documents/511/511_Trajectory_tracking/Yolo_result/" + camera + "/"
            output_name = file[:-3]
            results = model.track(source = input_file,
                                  #tracker='botsort.yaml',
                                  classes=[2],
                                  project = project_path,
                                  name= output_name,
                                  save=True,
                                  save_txt=True)

