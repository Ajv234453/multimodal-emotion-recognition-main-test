import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.svm import LinearSVC
# 指定视频所在的目录
video_directory = 'D:/DL/dirve_video/testdata/drive2'

label_01_videos = []
label_02_videos = []

# 遍历目录文件
for filename in os.listdir(video_directory):
    if not filename.endswith('.mp4'):  # 假设视频格式为mp4, 根据需要更改
        continue
    # 分割文件名以获取标签
    parts = filename.split('_')
    if len(parts) != 3:
        print(f"Unexpected filename format: {filename}")
        continue
    label = parts[1]
    # 根据标签将视频文件名添加到相应的列表
    if label == '01':
        label_01_videos.append(filename)
    elif label == '02':
        label_02_videos.append(filename)
    else:
        print(f"Unexpected label in filename: {filename}")

print("Label 01 videos:", label_01_videos)
print("Label 02 videos:", label_02_videos)


def extract_frames(video_path, num_frames=10):
    """
    提取视频帧。
    video_path: 视频路径
    num_frames: 要提取的帧数
    返回一个包含提取帧的numpy数组。
    """
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = total_frames // num_frames
    count = 0

    while count < total_frames:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        success, image = vidcap.read()
        if success:
            frames.append(image)
            count += frame_step
        else:
            break

    vidcap.release()
    return np.array(frames)

base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(video_frames):
    """
    使用ResNet50为视频帧提取特征。
    video_frames: numpy数组，包含视频帧。
    返回一个numpy数组，包含每个帧的特征。
    """
    preprocessed = preprocess_input(video_frames)
    return base_model.predict(preprocessed)



# 准备数据
X = []
y = []

for video_file in label_01_videos:
    frames = extract_frames(os.path.join(video_directory, video_file))
    features = extract_features(frames)
    # 这里，我们简单地平均所有帧的特征以获得一个特征向量
    avg_features = np.mean(features, axis=0)
    X.append(avg_features)
    y.append(0)  # 为label 01指定标签0

for video_file in label_02_videos:
    frames = extract_frames(os.path.join(video_directory, video_file))
    features = extract_features(frames)
    avg_features = np.mean(features, axis=0)
    X.append(avg_features)
    y.append(1)  # 为label 02指定标签1

# 使用SVM分类器进行分类
clf = LinearSVC()
clf.fit(X, y)


def predict_video(video_path, classifier):
    """
    预测给定视频的类别。
    video_path: 要预测的视频的路径
    classifier: 训练好的分类器
    返回预测的类别。
    """
    frames = extract_frames(video_path)
    features = extract_features(frames)
    avg_features = np.mean(features, axis=0)
    return classifier.predict([avg_features])[0]


# 指定要预测的视频的目录
test_videos_directory = 'D:/DL/dirve_video/testdata/drive2'

# 对指定目录中的每个视频进行预测
# for filename in os.listdir(test_videos_directory):
#     if not filename.endswith('.MP4'):
#         continue
#
#     video_path = os.path.join(test_videos_directory, filename)
#     prediction = predict_video(video_path, clf)
#
#     # 输出文件名和预测的类别
#     if prediction == 0:
#         print(f"{filename}: Label 01")
#     else:
#         print(f"{filename}: Label 02")

output_file_path = 'prediction_results_1.txt'

# 打开文件以写入结果
with open(output_file_path, 'w') as output_file:
    # 对指定目录中的每个视频进行预测
    for filename in os.listdir(test_videos_directory):
        if not filename.endswith('.mp4'):
            continue

        video_path = os.path.join(test_videos_directory, filename)
        prediction = predict_video(video_path, clf)

        # 格式化输出字符串
        output_line = f"File: {video_path} - Prediction: {prediction}\n"

        # 输出到控制台
        print(output_line, end='')

        # 写入到文件
        output_file.write(output_line)
