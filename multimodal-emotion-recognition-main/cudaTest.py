import torch
print(torch.cuda.is_available())  # 输出 True 表示CUDA可用
print(torch.version.cuda)         # 输出已安装的CUDA版本（如 12.1）