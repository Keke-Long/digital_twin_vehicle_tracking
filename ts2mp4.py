import os
import subprocess

# 每个ts文件转成单独的
# video_dir = "./Videos/"
# cameras = [folder for folder in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, folder))]
# for camera in sorted(cameras):
#     camera_dir = os.path.join(video_dir, camera)
#     for file in os.listdir(camera_dir):
#         if file.endswith(".ts"):
#             output_folder = camera_dir + "/mp4/"
#             os.makedirs(output_folder, exist_ok=True)
#
#             input_file = os.path.join(camera_dir, file)
#             output_file = os.path.join(output_folder, file[:-3] + ".mp4")
#
#             command = f'ffmpeg -i "{input_file}" -c:v copy -c:a copy "{output_file}"'
#             subprocess.call(command, shell=True)



# 所有ts文件转成一个mp4
video_dir = "./Videos/"
cameras = [folder for folder in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, folder))]
for camera in sorted(cameras):
    camera_dir = os.path.join(video_dir, camera)
    ts_files = [file for file in os.listdir(camera_dir) if file.endswith(".ts")]

    if len(ts_files) > 0:
        output_folder = os.path.join(camera_dir, "mp4")
        os.makedirs(output_folder, exist_ok=True)

        concat_file = os.path.join(camera_dir, "filelist.txt")

        # Create a file list with the paths of all ts files
        with open(concat_file, "w") as f:
            for ts_file in ts_files:
                f.write(f"file '{ts_file}'\n")

        output_file = os.path.join(output_folder, "output.mp4")

        # Concatenate ts files into one mp4 file using FFmpeg
        command = f'ffmpeg -f concat -safe 0 -i "{concat_file}" -c copy "{output_file}"'
        subprocess.call(command, shell=True)

        # Delete the file list
        os.remove(concat_file)
