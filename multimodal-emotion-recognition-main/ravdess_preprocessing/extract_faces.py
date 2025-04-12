import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化MTCNN
mtcnn = MTCNN(image_size=(720, 1280), device=device)

# 参数设置
save_frames = 15
input_fps = 30
save_length = 3.6  # seconds
save_avi = True
root = 'D:\CCFAcode\Video_Speech_Actor_24'
failed_videos = []

# 选择分布式帧的函数
select_distributed = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]

# 处理的视频计数
n_processed = 0

# 遍历root目录下的所有子目录
for subdir in tqdm(sorted(os.listdir(root))):
    subdir_path = os.path.join(root, subdir)
    if os.path.isdir(subdir_path):
        # 遍历子目录下的所有文件
        for filename in sorted(os.listdir(subdir_path)):
            if filename.endswith('.mp4'):
                # 检查对应的.npy文件是否已存在
                npy_filename = filename[:-4] + '_facecroppad.npy'
                npy_filepath = os.path.join(subdir_path, npy_filename)
                if os.path.exists(npy_filepath):
                    print(f"Skipping already processed file: {filename}")
                    continue  # 如果.npy文件已存在，则跳过当前文件的处理

                print(f"Processing file: {filename}")  # 打印当前正在处理的文件名
                cap = cv2.VideoCapture(os.path.join(subdir_path, filename))
                # 直接获取视频帧数
                framen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 选择要保存的帧
                frames_to_select = select_distributed(save_frames, min(framen, int(save_length * input_fps)))

                if framen > 0 and int(save_length * input_fps) > 0:
                    save_fps = save_frames // (min(framen, int(save_length * input_fps)) // input_fps)
                else:
                    print(f"Skipping file due to zero frame count or zero calculated frame count: {filename}")
                    continue  # 跳过当前文件的处理

                # 设置视频写入器
                if save_avi:
                    out = cv2.VideoWriter(os.path.join(subdir_path, filename[:-4] + '_facecroppad.avi'),
                                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), save_fps, (224, 224))

                numpy_video = []
                frame_ctr = 0


                # 逐帧读取视频
                while True:
                    ret, im = cap.read()
                    if not ret:
                        break
                    if frame_ctr in frames_to_select:
                        frames_to_select.remove(frame_ctr)

                        # 检查文件名是否以_r.mp4结尾
                        if not filename.endswith('_r.mp4'):
                            if im is None:  # 检查im是否为空
                                print(f"Warning: Frame is None at index {frame_ctr} in file {filename}")
                                continue  # 跳过当前循环的剩余部分
                            # 转换BGR到RGB
                            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            im_rgb = torch.tensor(im_rgb).to(device)

                            # 人脸检测
                            bbox, _ = mtcnn.detect(im_rgb)
                            if bbox is not None and len(bbox) > 0:
                                bbox = bbox[0]
                                x1, y1, x2, y2 = [int(b) for b in bbox]
                                # 裁剪人脸区域
                                im = im[y1:y2, x1:x2]
                            # else:
                            #     print(f"Invalid bbox at index {frame_ctr} in file {filename}")
                            #     im = None

                        # 调整帧大小
                        if im is None:  # 再次检查im是否为空
                            print(f"Warning: Frame is None after processing at index {frame_ctr} in file {filename}")
                            continue  # 跳过当前循环的剩余部分
                        im_resized = cv2.resize(im, (224, 224))
                        if save_avi:
                            out.write(im_resized)
                        numpy_video.append(im_resized)

                    frame_ctr += 1

                # 完成视频处理
                if save_avi:
                    out.release()
                np.save(os.path.join(subdir_path, filename[:-4] + '_facecroppad.npy'), np.array(numpy_video))

                if len(numpy_video) != save_frames:
                    print(f"Error processing file: {filename}")  # 指出处理失败的文件名
                    failed_videos.append(os.path.join(subdir, filename))

                n_processed += 1

# 记录处理结果
with open('processed.txt', 'a') as f:
    f.write(f"Processed {n_processed} files\n")
    f.write('\n'.join(failed_videos))

print(f"Failed videos: {failed_videos}")
