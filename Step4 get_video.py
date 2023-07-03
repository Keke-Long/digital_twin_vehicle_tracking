import os
import cv2
import math

def merge_images_to_video(camera_number, start_time, end_time):
    folder_path = f"./Track_result/{camera_number}/yolov8result/"  # Path to the camera folder
    output_path = f"output_{camera_number}_{start_time}_{end_time}.mp4"  # Output video path

    image_files = [file for file in os.listdir(folder_path) if file.endswith('.jpg')]  # Get files ending with .jpg

    # Filter image files within the specified start and end time range
    selected_files = []
    for file in image_files:
        file_name, ext = os.path.splitext(file)
        file_time = float(file_name)  # Get the standard time in seconds

        if start_time <= file_time <= end_time:
            selected_files.append(file)

    # Sort selected image files by filename to ensure chronological order in the video
    selected_files.sort(key=lambda x: float(os.path.splitext(x)[0]))

    # Get width and height information from the first image
    first_image_path = os.path.join(folder_path, selected_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Calculate the frame rate and corresponding start and end frames
    frame_rate = 5  # Generate 5 frames per second
    start_frame = math.floor(start_time * frame_rate)
    end_frame = math.ceil(end_time * frame_rate)

    # Create a video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (width, height))

    # Generate the video frame by frame
    for i in range(start_frame, end_frame + 1):
        time = i / frame_rate  # Current time corresponding to the frame
        nearest_time_file = min(selected_files, key=lambda x: abs(float(os.path.splitext(x)[0]) - time))
        image_path = os.path.join(folder_path, nearest_time_file)
        image = cv2.imread(image_path)
        out.write(image)

    # Release resources
    out.release()
    print("Video merge completed!")

# Usage example
camera_number = "0025"  # Replace with the actual camera number
start_time = 1685917196  # Replace with the actual start time
end_time = start_time+10  # Replace with the actual end time
merge_images_to_video(camera_number, start_time, end_time)