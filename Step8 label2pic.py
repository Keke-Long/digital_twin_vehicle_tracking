'''
optional
If you don't want to save a video with tracking result, you can skip this step
Restore the labeled data to the original image
将标注数据还原到原图上
'''


import os
import pandas as pd
import numpy as np
import cv2
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

# Extract float from filename
def extract_float(filename):
    base = os.path.basename(filename)  # Get the filename without path
    float_number = os.path.splitext(base)[0]  # Remove the extension
    try:
        return float(float_number)
    except ValueError:
        raise ValueError(f"Filename {filename} does not contain a valid float number.")

def load_speed_data(csv_path): # Create a lookup table
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
    # Generate color using the normalized speed value
    # Here we use the HSV color space, where the H value (hue) changes with speed
    color_hue = (1.0 - normalized_speed) * 120  # H value ranges from red (speed=0) to green (speed=max_speed)
    color = cv2.cvtColor(np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color)

def xywh2xyxy(x, w1, h1, img, timestamp): # Coordinate conversion
    if len(x) == 5:
        label, x, y, w, h = x
        id = 0
    elif len(x) == 6:  # If there is an id, unpack normally
        label, x, y, w, h, id = x
    # print("Original width and height:\nw1={}\nh1={}".format(w1, h1))
    # Bounding box de-normalization
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1

    # Calculate coordinates
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    # Find speed
    speed = 30
    try:
        # Ensure the precision of timestamp matches frame_num
        rounded_timestamp = round(timestamp, 2)
        rounded_id = int(id)
        # print("(rounded_id, rounded_timestamp)", (rounded_id, rounded_timestamp))
        speed = speed_lookup.loc[(rounded_id, rounded_timestamp), 'speed']
        # If the obtained speed is Series (possibly because of multiple identical indices), take the first one
        if isinstance(speed, pd.Series):
            speed = speed.iloc[0]
    except KeyError:
        # If direct access fails, try to find the nearest timestamp
        search_times = np.arange(rounded_timestamp - 1, rounded_timestamp + 4, 0.01)
        for time in search_times:
            try:
                rounded_time = round(time, 2)
                if (rounded_id, rounded_time) in speed_lookup.index:
                    speed = speed_lookup.loc[(rounded_id, rounded_time), 'speed']
                    # If the obtained speed is Series, take the first one
                    if isinstance(speed, pd.Series):
                        speed = speed.iloc[0]
                    break  # If speed is found, exit the loop
            except KeyError:
                continue  # If speed is not found at this timestamp, continue looping

    # Check if speed is NaN or zero
    if speed is None or pd.isna(speed) or speed == 0:
        speed = 30
    # print('speed', speed)

    # Get color based on speed
    color = speed_to_color(speed, 40) if speed is not None else (255, 255, 255)

    # Draw rectangle
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color, 2)
    text_position = (int(x * w1 - w * w1 / 2), int(y * h1 - h * h1 / 2) - 10)
    cv2.putText(img, f"{speed:.2f}m/s", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

if __name__ == '__main__':
    # Modify input image folder
    img_folder = "./frames/case1/"
    # Modify input label folder
    label_folder = "./Yolo_result/case1/labels/"
    # Output image folder location
    output_folder = "/home/ubuntu/Documents/511/511_Trajectory_tracking/Step8 case1/"
    # Read speed data
    speed_lookup = load_speed_data("Trajectory/case1_after step 7.csv")

    # Calculate the minimum timestamp
    min_timestamp = min(get_timestamp_from_filename(f"{i}.jpg") for i in range(1501))

    for label_num in range(1501):
        # Construct file paths for labels and images
        label_path = os.path.join(label_folder, f"{label_num}.txt")
        image_path = os.path.join(img_folder, f"{label_num}.jpg")

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file not found for: {label_path}, skipping.")
            continue

        timestamp = round(label_num / 15, 2)

        # Read image file
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            # Read labels
            with open(label_path, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            # Draw each target
            for x in lb:
                # De-normalize and get top-left and bottom-right coordinates, draw rectangle
                img = xywh2xyxy(x, w, h, img, round(timestamp, 2))
            # cv2.imshow('show', img)
            cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)
