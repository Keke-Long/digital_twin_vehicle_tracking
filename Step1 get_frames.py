import os
import cv2
import re


def extract_frames_from_video(video_path, output_dir):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频文件的帧速率和总帧数
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 检查帧速率是否不合理，若不合理则设置为15
    if fps < 1 or fps > 100:
        fps = 15  # 设置为15帧/秒

    # 获取视频的结束时间戳（秒）
    #video_end_time = int(re.split(r"[_\.]", os.path.basename(video_path))[-2])
    video_end_time = 1685917912

    # 设置帧提取的间隔为1帧
    frame_interval = 1

    print(f"视频文件: {video_path}，帧速率: {fps}，总帧数: {total_frames}")

    # 逐帧提取并保存为图像
    frame_idx = 0
    ret, frame = cap.read()
    frame_timestamp = 0
    while ret:
        # 每帧都保存
        # frame_timestamp = video_end_time - total_frames/fps + frame_idx / fps
        frame_timestamp += 1
        image_path = os.path.join(output_dir, f"{int(frame_timestamp)}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"保存帧 {image_path}")

        # 读取下一帧
        ret, frame = cap.read()

        # 更新帧索引
        frame_idx += 1

    # 释放视频文件句柄
    cap.release()




# video_dir = "./Videos/"
# cameras = [folder for folder in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, folder))]
#
# # 选择处理特定的摄像头文件夹
# selected_cameras = ['0001']  # 举例选择了camera_01和camera_02
# for camera in selected_cameras:
#     camera_dir = os.path.join(video_dir, camera)
#     video_files = [file for file in os.listdir(camera_dir) if file.endswith(".ts")]
#
#     # 创建保存帧的目录
#     output_dir = os.path.join('./frames/', camera.split("_")[-1])
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 选择要处理的特定视频文件
#     selected_videos = video_files
#     for video_file in selected_videos:
#         video_path = os.path.join(camera_dir, video_file)
#         extract_frames_from_video(video_path, output_dir)


# 直接给出MP4视频的路径
video_path = "/home/ubuntu/Documents/511/511_Trajectory_tracking/Videos/0001/mp4/output.mp4"  # 使用您的视频文件路径替换这里

# 确定输出目录，您可以根据视频名或其他逻辑命名
output_dir = "/home/ubuntu/Documents/511/511_Trajectory_tracking/frames/0001/"
os.makedirs(output_dir, exist_ok=True)

# 调用函数处理视频并提取所有帧
extract_frames_from_video(video_path, output_dir)