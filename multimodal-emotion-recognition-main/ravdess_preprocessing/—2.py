# -*- coding: utf-8 -*-
"""
This code is base on https://github.com/okankop/Efficient-3DCNNs
"""

import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa


# 加载 NumPy 格式的视频文件，并将其转换为 PIL 图像列表。
def video_loader(video_dir_path):
    video = np.load(video_dir_path)
    video_data = []
    for i in range(np.shape(video)[0]):
        video_data.append(Image.fromarray(video[i, :, :, :]))
    return video_data


def get_default_video_loader():
    return functools.partial(video_loader)


# 使用 librosa 加载音频文件
def load_audio(audiofile, sr):
    audios = librosa.core.load(audiofile, sr)
    y = audios[0]
    return y, sr


def road_loader(road_path):
    # road = np.load(road_data_path)
    # road_data = [Image.fromarray(frame) for frame in road_data]
    road = np.load(road_path)
    road_data = []
    for i in range(np.shape(road)[0]):
        road_data.append(Image.fromarray(road[i, :, :, :]))
    return road_data


# 提取 MFCC 特征
def get_mfccs(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=10)
    return mfcc


# 从注释文件中构建一个数据集。
def make_dataset(subset, annotation_path):
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
    # 1. 检查注释文件是否为空
    if not annots:
        print("Warning: Annotation file is empty!")
        return []

    # 2. 验证注释文件的格式
    for line in annots:
        parts = line.split(';')
        if len(parts) != 5:
            print(f"Warning: Incorrect format for line: {line}")
            return []

    # 3. 检查subset参数与注释文件中的内容
    unique_subsets = set([line.split(';')[-1].rstrip() for line in annots])
    print(f"Available subsets in annotations: {unique_subsets}")
    if subset not in unique_subsets:
        print(f"Warning: The provided subset '{subset}' does not exist in the annotations.")
        return []

    dataset = []
    for line in annots:
        filename, audiofilename, roadfilename, label, trainvaltest = line.split(';')
        if trainvaltest.rstrip() != subset:
            continue

        sample = {'video_path': filename,
                  'audio_path': audiofilename,
                  'road_path': roadfilename,
                  'label': int(label) - 1}
        dataset.append(sample)
    return dataset


# 加载视频、音频或二者结合的数据。
class RAVDESS(data.Dataset):
    def __init__(self,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 get_loader=get_default_video_loader, data_type='audiovisualroad', audio_transform=None):
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.road_loader = road_loader
        self.data_type = data_type
        # print("Dataset from make_dataset:")
        # print(self.data)
        # print("Annotation Path:", annotation_path)

    def __getitem__(self, index):
        default_video = torch.zeros(3, 10, 224, 224)  # 假设视频数据是3通道，10帧，每帧大小为224x224
        default_audio = torch.zeros(10, 20)  # 假设音频特征是10个MFCC系数，每个系数有20个时间步
        default_road = torch.zeros(3, 10, 224, 224)  # 假设道路数据格式与视频相同
        clip = default_video
        audio_features = default_audio
        road_frames = default_road
        try:
            # data_item = self.data[index]
            # print(f"Data item at index {index}: {data_item}")  # 打印当前索引处的数据项

            target = self.data[index]['label']
            if self.data_type == 'video' or self.data_type == 'audiovisualroad':
                path = self.data[index]['video_path']

                clip = self.loader(path)
                # print("Video Path:", path)
                if self.spatial_transform is not None and clip is not None:
                    self.spatial_transform.randomize_parameters()
                    clip = [self.spatial_transform(img) for img in clip]
                    clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
                else:
                    print("vnone")
                    clip = default_video

                if self.data_type == 'video':
                    return clip, target

            if self.data_type == 'audio' or self.data_type == 'audiovisualroad':
                path = self.data[index]['audio_path']
                print(path)
                y, sr = load_audio(path, sr=22050)
                if y is not None:
                    if self.audio_transform is not None:
                        y = self.audio_transform(y)
                    audio_features = get_mfccs(y, sr)
                else:
                    print("anone")
                    audio_features = default_audio

                if self.data_type == 'audio':
                    return audio_features, target

            if self.data_type == 'road' or self.data_type == 'audiovisualroad':
                road_path = self.data[index]['road_path']
                print(road_path)
                road_frames = self.road_loader(road_path)
                if self.spatial_transform is not None and road_frames is not None:
                    road_frames = [self.spatial_transform(frame) for frame in road_frames]
                    road_frames = torch.stack(road_frames, 0).permute(1, 0, 2, 3)
                else:
                    print("rnone")
                    road_frames = default_road

            if self.data_type == 'audiovisualroad':
                return audio_features, clip, road_frames, target

            print(f"Before accessing problematic part at index {index}")
            # 可能出问题的代码行
            print(f"After accessing problematic part at index {index}")
        except Exception as e:
            print(f"Error processing item at index {index}: {e}")

    def __len__(self):
        return len(self.data)

