import os
import subprocess

video_dir = "./Videos/"
cameras = [folder for folder in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, folder))]
for camera in sorted(cameras):
    camera_dir = os.path.join(video_dir, camera)
    for file in os.listdir(camera_dir):
        if file.endswith(".ts"):
            output_folder = camera_dir + "/mp4/"
            os.makedirs(output_folder, exist_ok=True)

            input_file = os.path.join(camera_dir, file)
            output_file = os.path.join(output_folder, file[:-3] + ".mp4")

            command = f'ffmpeg -i "{input_file}" -c:v copy -c:a copy "{output_file}"'
            subprocess.call(command, shell=True)

