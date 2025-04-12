
import re

# 定义权重
weights = {'file1': 0.6, 'file2': 0.4}

def parse_file_content_file1(file_content):
    result = {}
    for line in file_content.strip().split('\n'):
        match = re.search(r'\\([^\\]+)_facecroppad.npy - Prediction: (\d+)', line)
        if match:
            filename, prediction = match.groups()
            result[filename] = int(prediction)
    return result

def parse_file_content_file2(file_content):
    result = {}
    for line in file_content.strip().split('\n'):
        match = re.search(r'\\([^\\]+)\.mp4 - Prediction: (\d+)', line)
        if match:
            filename, prediction = match.groups()
            result[filename] = int(prediction)
    return result

# 读取文件内容
with open('D:/DL/model/2.multimodal-emotion-recognition-main/multimodal-emotion-recognition-main/results/prediction.txt', 'r') as f:
    file1_content = f.read()
with open('prediction_results.txt', 'r') as f:
    file2_content = f.read()
# 解析文件内容
predictions_file1 = parse_file_content_file1(file1_content)
predictions_file2 = parse_file_content_file2(file2_content)

# 计算带权均值并将结果写入新文件
with open('result.txt', 'w') as f:
    for filename in predictions_file1:
        if filename in predictions_file2:
            weighted_avg = (predictions_file1[filename] * weights['file1'] +
                            predictions_file2[filename] * weights['file2'])
            prediction = 1 if weighted_avg > 0 else 0
            f.write(f'File: {filename} - Weighted Prediction: {prediction}\n')


