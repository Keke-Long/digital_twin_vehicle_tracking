# Vehicle Trajectory Tracking

This project is a simplified version of Project A. When you have data captured from fixed cameras, this project allows you to obtain trajectories in the world coordinate system. Yolov8 is required in this project.


## Steps to Follow

Please follow the steps from Step1 to Step9. Note that Step4 and Step789 are for visualizing the results. If you only need a trajectory CSV file, you can skip these steps.

### Step1: Extract Frames from Video
Extracts frames from the input video at a specified frame rate.

### Step2: Apply Super-Resolution
Applies super-resolution to the extracted frames to enhance image quality.

### Step3: Object Detection and Tracking
Uses YOLOv8 for object detection and tracking on the extracted frames.

### Step4: Generate Video with Tracking Results
Combines the frames with tracking results into a video for visualization. **(Optional)**

### Step5: Convert Tracking Results to Trajectories
Converts the tracking results to a trajectory CSV file.

### Step6: Convert Pixel Coordinates to World Coordinates
Converts the pixel coordinates from the tracking results to world coordinates using camera calibration data.

### Step7: Post-Processing of Trajectories
Post-processes the trajectory data for further analysis.

### Step8: Visualize Labels on Original Images
Restores labeled data onto the original images for visualization. **(Optional)**

### Step9: Combine Images into a Video
Combines all the labeled images into a single MP4 file. **(Optional)**


## For UAV or Moving Camera Data

If you are using data captured from UAVs or other moving cameras, it is recommended to use a more complex repository: [ANL_ParkSt_trj_data_processing](https://github.com/Keke-Long/ANL_ParkSt_trj_data_processing)




