'''
检查视频帧率，确定自己要处理的帧率，然后把视频需要的关键帧保存为图片
把视频拆成每一帧而不是直接用视频的原因：
    1方便项目涉及多个视频且视频帧率不统一
    2基于yolov8时，用图片效果比视频作为input好

Check the video frame rate, determine the frame rate you want to process, and then save the key frames as images
Reasons for splitting the video into each frame instead of using the video directly:
    1 It is convenient for projects involving multiple videos and the video frame rates are not uniform
    2 When based on yolov8, using images is better in tracking than using videos as input
'''


import os
import cv2


def check_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return

    # Get video file properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Print video properties
    print(f"Video file: {video_path}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"FPS: {fps}")
    cap.release()


def extract_frames_from_video(video_path, output_dir, target_fps):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(f"Unable to open video file: {video_path}")
        return

    # Get video file frame rate and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate frame extraction interval
    frame_interval = int(fps / target_fps)

    print(f"Video file: {video_path}, FPS: {fps}, Total frames: {total_frames}")

    # Extract and save frames
    frame_idx = 0
    saved_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            if frame_idx >= total_frames:
                print("Video reading complete")
                break
            else:
                print(f"Skipping corrupted frame {frame_idx}")
                frame_idx += 1
                continue

        # Save frame at every frame_interval
        if frame_idx % frame_interval == 0:
            image_path = os.path.join(output_dir, f"{saved_frame_idx:05d}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Saved frame {image_path}")
            saved_frame_idx += 1

        frame_idx += 1

    # Release video file handle
    cap.release()



# Path to the MP4 video
video_path = "/home/ubuntu/Documents/511/John_Nolan_dr_151.mp4"

# Call the function to check video properties
check_video_properties(video_path)

# Determine output directory, you can name it based on the video name or other logic
output_dir = "./step1_output/video1"
os.makedirs(output_dir, exist_ok=True)

# Call the function to process the video and extract frames at the specified frame rate
extract_frames_from_video(video_path, output_dir, target_fps=5)
