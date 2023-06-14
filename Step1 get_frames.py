import os
import cv2

def extract_frames_from_videos(video_dir):
    # 获取视频目录下所有文件夹（摄像头）
    cameras = [folder for folder in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, folder))]

    # 遍历摄像头文件夹
    for camera in sorted(cameras):
        camera_dir = os.path.join(video_dir, camera)
        # 获取摄像头文件夹下的视频文件
        video_files = [file for file in os.listdir(camera_dir) if file.endswith(".ts")]

        # 获取摄像头编号
        camera_number = camera.split("_")[-1]

        # 创建保存帧的目录
        output_dir = os.path.join('./frames/', camera_number)
        os.makedirs(output_dir, exist_ok=True)

        # 遍历视频文件
        for video_file in sorted(video_files)[:5]:  # 仅处理前5个视频
            video_path = os.path.join(camera_dir, video_file)

            # 打开视频文件
            cap = cv2.VideoCapture(video_path)

            # 检查视频文件是否成功打开
            if not cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                continue

            # 获取视频文件的帧速率和总帧数
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 检查帧速率是否不合理，若不合理则设置为15
            if fps < 1 or fps > 100:
                fps = 15  # 设置为15帧/秒

            # 计算视频的总时长（秒）
            total_video_time = total_frames / fps

            # 重新定位到视频的起始位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # 获取视频的结束时间戳（秒）
            video_end_time = int(video_file.split("_")[-2])
            print(f"视频文件: {video_path}，结束时间戳: {video_end_time}")

            # 获取第一帧的时间戳
            ret, frame = cap.read()
            if not ret:
                print(f"无法读取视频文件的第一帧: {video_path}")
                cap.release()
                continue
            timestamp = 0

            # 设置帧提取的间隔为3帧
            frame_interval = 3

            print(f"视频文件: {video_path}，帧速率: {fps}，总帧数: {total_frames}")

            # 逐帧提取并保存为图像
            frame_idx = 0
            while ret:
                # 如果是每隔3帧，则保存帧
                if frame_idx % frame_interval == 0:
                    # 计算帧的时间戳（秒）
                    frame_timestamp = video_end_time + round((frame_idx * (1 / fps)), 2)

                    # 将帧保存为JPEG图像
                    image_path = os.path.join(output_dir, f"{frame_timestamp:.2f}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"保存帧 {image_path}")

                # 读取下一帧
                ret, frame = cap.read()

                # 更新帧索引
                frame_idx += 1

            # 释放视频文件句柄
            cap.release()


video_dir = "./Videos0065/"
extract_frames_from_videos(video_dir)