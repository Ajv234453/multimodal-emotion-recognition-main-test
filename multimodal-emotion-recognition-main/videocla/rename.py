import os

# 指定文件所在的目录
directory_path = 'D:/DL/Video_Speech_Actor_22'  # 请更改为你的文件夹路径

# 计数器，用于生成文件名
counter = 1

# 为目录中的每个文件生成新的文件名并重命名
for filename in sorted(os.listdir(directory_path)):
    # 生成新的文件名
    new_filename = f"01_01_01{counter}.MP4"  # 更改extension为实际的文件扩展名, 如: mp4

    # 获取原始文件和新文件的完整路径
    old_path = os.path.join(directory_path, filename)
    new_path = os.path.join(directory_path, new_filename)

    # 重命名文件
    os.rename(old_path, new_path)

    # 更新计数器
    counter += 1

print("Files renamed successfully!")
