# -*- coding: utf-8 -*-

import librosa
import os
import soundfile as sf
import numpy as np
import time  # 用于计算耗时

#audiofile = 'E://OpenDR_datasets//RAVDESS//Actor_19//03-01-07-02-01-02-19.wav'
##this file preprocess audio files to ensure they are of the same length. if length is less than 3.6 seconds, it is padded with zeros in the end. otherwise, it is equally cropped from 
##both sides

root = r'D:\CCFAcode\Video_Speech_Actor_24'
target_time = 3.6  # sec

# ---- 新增部分：初始化计数器和时间 ----
total_files = 0
processed_files = 0
start_time = time.time()

# 第一次遍历：统计总文件数（用于进度提示）
for actor in os.listdir(root):
    actor_dir = os.path.join(root, actor)
    if os.path.isdir(actor_dir):  # 确保是文件夹
        for audiofile in os.listdir(actor_dir):
            if audiofile.endswith('.wav') and 'croppad' not in audiofile:
                total_files += 1

print(f"发现需处理的音频文件总数: {total_files} 个")

# 第二次遍历：实际处理
for actor in os.listdir(root):
    actor_dir = os.path.join(root, actor)
    if not os.path.isdir(actor_dir):  # 跳过非文件夹
        continue

    for audiofile in os.listdir(actor_dir):
        if not audiofile.endswith('.wav') or 'croppad' in audiofile:
            continue

        # ---- 处理逻辑 ----
        audios = librosa.core.load(os.path.join(actor_dir, audiofile), sr=22050)
        y, sr = audios[0], audios[1]
        target_length = int(sr * target_time)

        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')  # 更高效的补零方式
        else:
            remain = len(y) - target_length
            y = y[remain // 2: -(remain - remain // 2)]

        sf.write(os.path.join(actor_dir, audiofile[:-4] + '_croppad.wav'), y, sr)

        # ---- 新增部分：进度提示 ----
        processed_files += 1
        if processed_files % 10 == 0:  # 每处理10个文件打印一次进度
            print(f"已处理 {processed_files}/{total_files} 个文件，耗时 {time.time() - start_time:.1f} 秒")

# ---- 新增部分：最终完成提示 ----
print(f"\n✅ 全部完成！共处理 {processed_files} 个文件，总耗时 {time.time() - start_time:.1f} 秒")
