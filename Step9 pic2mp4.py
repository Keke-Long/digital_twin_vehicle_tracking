'''
将一个文件夹下的所有 PNG 图像合并成一个 MP4 文件
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


# 文件夹路径
folder_path = './Step8 results'
# 输出视频的路径
video_path = './Step9 results/0001.mp4'
# 帧率
fps = 15

# 获取文件夹中的所有图像文件名
images = [img for img in os.listdir(folder_path) if img.endswith(".png")]
# 按文件名排序，这里假设文件名中包含排序信息
images.sort(key=natural_keys)

# 获取第一个图像的尺寸
frame = cv2.imread(os.path.join(folder_path, images[0]))
height, width, layers = frame.shape

# 定义视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者 'XVID'
video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(folder_path, image)))

#cv2.destroyAllWindows()
video.release()
