'''
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

# 提取文件名中的浮点数
def extract_float(filename):
    base = os.path.basename(filename)  # 获取文件名，去掉路径
    float_number = os.path.splitext(base)[0]  # 去掉扩展名
    try:
        return float(float_number)
    except ValueError:
        raise ValueError(f"Filename {filename} does not contain a valid float number.")


def load_speed_data(csv_path): # 创建一个查询表
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
    # 使用归一化的速度值生成颜色
    # 这里使用HSV颜色空间，其中H值（色调）随速度变化
    color_hue = (1.0 - normalized_speed) * 120  # H值范围从红色（速度=0）到绿色（速度=max_speed）
    color = cv2.cvtColor(np.uint8([[[color_hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return tuple(int(c) for c in color)


def xywh2xyxy(x, w1, h1, img, timestamp): # 坐标转换
    if len(x) == 5:
        label, x, y, w, h = x
        id = 0
    elif len(x) == 6:  # 如果有id，则正常解包
        label, x, y, w, h, id = x
    # print("原图宽高:\nw1={}\nh1={}".format(w1, h1))
    # 边界框反归一化
    x_t = x * w1
    y_t = y * h1
    w_t = w * w1
    h_t = h * h1

    # 计算坐标
    top_left_x = x_t - w_t / 2
    top_left_y = y_t - h_t / 2
    bottom_right_x = x_t + w_t / 2
    bottom_right_y = y_t + h_t / 2

    # 寻找速度
    speed = 30
    try:
        # 确保timestamp的精度与frame_num相匹配
        rounded_timestamp = round(timestamp, 2)
        rounded_id = int(id)
        # print("(rounded_id, rounded_timestamp)", (rounded_id, rounded_timestamp))
        speed = speed_lookup.loc[(rounded_id, rounded_timestamp), 'speed']
        # 如果获取到的速度是Series（可能是因为有多个相同的索引），则取第一个
        if isinstance(speed, pd.Series):
            speed = speed.iloc[0]
    except KeyError:
        # 如果直接获取失败，尝试找最近的时间戳
        search_times = np.arange(rounded_timestamp - 1, rounded_timestamp + 4, 0.01)
        for time in search_times:
            try:
                rounded_time = round(time, 2)
                if (rounded_id, rounded_time) in speed_lookup.index:
                    speed = speed_lookup.loc[(rounded_id, rounded_time), 'speed']
                    # 如果获取到的速度是Series，则取第一个
                    if isinstance(speed, pd.Series):
                        speed = speed.iloc[0]
                    break  # 如果找到速度，退出循环
            except KeyError:
                continue  # 如果在这个时间戳没有找到速度，继续循环

    # 检查速度是否是NaN或为0
    if speed is None or pd.isna(speed) or speed == 0:
        speed = 30
    # print('speed', speed)

    # 根据速度获取颜色
    color = speed_to_color(speed, 40) if speed is not None else (255, 255, 255)

    # 绘制矩形框
    cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), color, 2)
    text_position = (int(x * w1 - w * w1 / 2), int(y * h1 - h * h1 / 2) - 10)
    cv2.putText(img, f"{speed:.2f}m/s", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img



if __name__ == '__main__':
    # 修改输入图片文件夹
    img_folder = "./frames/0001/"
    # 修改输入标签文件夹
    label_folder = "./Yolo_result/0001_11.63/labels/"
    # 输出图片文件夹位置
    output_folder = "/home/ubuntu/Documents/511/511_Trajectory_tracking/Step8 results/"
    # 读取速度数据
    speed_lookup = load_speed_data(f"Trajectory/0001_after step 7.csv")

    # 计算最小时间戳
    min_timestamp = min(get_timestamp_from_filename(f"{i}.jpg") for i in range(1501))

    for label_num in range(1501):
        # 构建标签和图像的文件路径
        label_path = os.path.join(label_folder, f"output_{label_num}.txt")
        image_path = os.path.join(img_folder, f"{label_num}.jpg")
        print('image', f"{label_num}.jpg", 'label', f"output_{label_num}.txt")

        # Check if the label file exists
        if not os.path.exists(label_path):
            print(f"Label file not found for: {label_path}, skipping.")
            continue

        timestamp = round(label_num / 15, 2)

        # 读取图像文件
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            # 读取 labels
            with open(label_path, 'r') as f:
                lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
            # 绘制每一个目标
            for x in lb:
                # 反归一化并得到左上和右下坐标，画出矩形框
                img = xywh2xyxy(x, w, h, img, round(timestamp,2))
            #cv2.imshow('show', img)
            cv2.imwrite(output_folder + '/' + '{}.png'.format(image_path.split('/')[-1][:-4]), img)