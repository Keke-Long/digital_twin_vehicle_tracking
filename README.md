Digital twin vehicle tracking aims to track vehicles on the video with object detection algorithm and trajectory calculation algorithms. Our algorithm will add the bounding box and current speed of vehicles in the video.

![0001](0001.gif)


# Usage

First, create your virtual environment.
```python
conda create -n {Your Project Name} python=3.10
conda activate {Your Project Name}
pip install -r requirements.txt
```

Download your prefered pretrained YOLO from [Ultralytics webpage]( https://github.com/ultralytics/ultralytics). We used YOLOv8n for our test.

The repository structures are these.

```bash
├── data/
│   ├── videos/
├── results/
│   ├── frames/
│       │── video_id/
│   ├── labeled_results/
│       │── video_id/
│   ├── trajectory/
│   └── yolo_results/
│       └── video_id/
├── main.py
├── utils.py
├── ts2mp4.py
├── requirements.txt
└── .gitignore
  
```

You can turn on and off the flag as which process you want to run.

```python
flg_dict = {'is_yolo': False, 'is_merge_images': False, 'is_traj': False, 'is_calibration': False, 
            'is_process_traj': True, 'is_label_to_pic': True, 'is_pic_to_video': True}
```

Then run the code by ```python main.py```.

# Lincense


