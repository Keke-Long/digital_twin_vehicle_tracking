'''
optional
If you don't want to save a video with tracking result, you can skip this step
Combine all PNG images in a folder into one MP4 file
'''

import cv2
import os
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

# Folder path
folder_path = './Step8 case1'
# Output video path
video_path = './Step9 results/case1.mp4'
# Frame rate
fps = 15

# Get all image filenames in the folder
images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
# Sort filenames, assuming they contain sorting information
images.sort(key=natural_keys)

# Get the dimensions of the first image
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

# Define the video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(folder_path, image)))

# cv2.destroyAllWindows()
video.release()
