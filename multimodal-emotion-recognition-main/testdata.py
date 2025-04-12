import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler


from model import generate_model
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time
from opts import parse_opts
opt = parse_opts()
training_data = get_test_set(opt, spatial_transform=None)
# 获取训练数据
train_loader = torch.utils.data.DataLoader(
    training_data,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True)
for i, data in enumerate(train_loader):
    print(i)