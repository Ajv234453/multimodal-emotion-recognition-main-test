import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import os
#use mlkag
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
predictions_model1 = parse_file_content_file1(file1_content)
predictions_model2= parse_file_content_file2(file2_content)

print(predictions_model1)
print(predictions_model2)
# 获取交集中的文件名，确保两个模型都对这些文件进行了预测
common_files = set(predictions_model1.keys()) & set(predictions_model2.keys())

# 结构化数据
X_stacked = np.array([[predictions_model1[file], predictions_model2[file]] for file in common_files])
#读取标签
video_directory = 'D:/DL/dirve_video/testdata/drive2'
file_names = []
labels = []

for filename in os.listdir(video_directory):
    if filename.endswith('.mp4'):
        parts = filename.split('_')
        if len(parts) == 3:
            label = int(parts[1]) - 1  # 01对应标签0，02对应标签1
            file_names.append(parts[0] + '_' + parts[1] + '_' + parts[2].split('.')[0])  # 提取文件名，例如 '01_01_1'
            labels.append(label)
        else:
            print(f"Unexpected filename format: {filename}")

y_stacked = np.array(labels)
#meta_model = RandomForestClassifier()
#meta_model = LogisticRegression()
meta_model = GradientBoostingClassifier()
meta_model.fit(X_stacked, y_stacked)
final_predictions = meta_model.predict(X_stacked)
final_predictions_dict = {file: prediction for file, prediction in zip(common_files, final_predictions)}

from sklearn.metrics import accuracy_score

# 计算准确率
accuracy = accuracy_score(y_stacked, final_predictions)
print("Accuracy:", accuracy)

# 将预测结果写入文件
with open('stacking_GradientBoostingClassifier_prediction.txt', 'w') as f:
    for file_name, prediction in final_predictions_dict.items():
        f.write(f"File: {file_name} - Prediction: {prediction}\n")

print("Stacking predictions have been written to stacking_prediction.txt")
