import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger, adjust_learning_rate, save_checkpoint
from train import train_epoch
from validation import val_epoch
import time
# Load model weights
opt = parse_opts()
model, _ = generate_model(opt)  # Generate the model architecture
checkpoint_path = os.path.join(opt.result_path, 'RAVDESS_multimodalcnn_15_best0.pth')  # Replace with your model filename
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model.to(opt.device)
model.eval()  # Set the model to evaluation mode
video_transform = transforms.Compose([
    transforms.ToTensor(opt.video_norm_value)
])

test_data = get_test_set(opt, spatial_transform=video_transform)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=opt.n_threads,
    pin_memory=True
)
all_predictions = []
with torch.no_grad():  # Disable gradient calculation for efficiency
    for inputs, _ in test_loader:
        inputs = inputs.to(opt.device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_predictions.extend(predicted.cpu().numpy())


