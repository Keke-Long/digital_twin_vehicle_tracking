'''
object dectection
Input: Keyframes
method: yolov8
'''


from ultralytics import YOLO
import os


input_file = "./step1_output/video1/"

model = YOLO('yolov8l.pt') # use large model, if you don't care about the running time
project_path = "./step3_Yolo_result/"
output_name = 'video1'
model.track(source = input_file,
              tracker='bytetrack.yaml',
              classes=[0, 1, 2, 3, 6, 8, 18],
              project = project_path,
              name= output_name,
              save=True,
              conf=0.1,
              iou = 0.6,
              imgsz = 1920, # use the original image size, if you don't care about the running time
              augment=True,line_thickness=3,hide_conf=True,
              save_txt=True)