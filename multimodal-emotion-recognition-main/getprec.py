import os
import torch
from torch import nn
from model import generate_model
import transforms
from dataset import get_training_set, get_validation_set, get_test_set
from opts import parse_opts
from validation import val_epoch
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transforms")

def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]  # remove 'module.' prefix
        new_state_dict[name] = v
    return new_state_dict

def predict_with_pretrained_model():
    opt = parse_opts()
    # 设定设备（CPU或CUDA）
    if opt.device != 'cpu' and torch.cuda.is_available():
        opt.device = 'cuda'
    else:
        opt.device = 'cpu'
    # 加载模型
    model, _ = generate_model(opt)
    if isinstance(model, nn.DataParallel):
        model = model.module
    best_state = torch.load(os.path.join(opt.result_path, f'RAVDESS_multimodalcnn_15_best0.pth'))
    model.load_state_dict(fix_state_dict(best_state['state_dict']))
    model.to(opt.device)
    model.eval()
    # 定义损失函数
    criterion = nn.CrossEntropyLoss().to(opt.device)
    # 准备测试数据
    video_transform = transforms.Compose([
        transforms.ToTensor(opt.video_norm_value)
    ])
    test_data = get_validation_set(opt, spatial_transform=video_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True
    )
    for data in test_loader:
        print("Number of tensors returned:", len(data))
        break
    # 进行预测
    predictions = []
    file_paths = []
    for audio_inputs, inputs_visual, targets in test_loader:
        # Only get paths for the current batch
        batch_size = audio_inputs.size(0)
        start_idx = len(predictions)  # the current number of predictions can serve as the starting index for this batch
        end_idx = start_idx + batch_size
        current_file_paths = [sample['video_path'] for sample in test_loader.dataset.data[start_idx:end_idx]]
        file_paths.extend(current_file_paths)

        inputs_visual = inputs_visual.permute(0, 2, 1, 3, 4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0] * inputs_visual.shape[1], inputs_visual.shape[2],
                                              inputs_visual.shape[3], inputs_visual.shape[4])

        with torch.no_grad():
            audio_inputs = audio_inputs.to(opt.device)
            inputs_visual = inputs_visual.to(opt.device)
            outputs = model(audio_inputs, inputs_visual)
            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.squeeze(1)  # remove the second dimension to get predictions of shape [batch_size]
            predictions.extend(pred.cpu().numpy().tolist())

    for fpath, pred in zip(file_paths, predictions):
        print(f"File: {fpath} - Prediction: {pred}")
    # ... rest of the code ...

    # 输出预测结果
    #print(predictions)
    with open('./results/prediction.txt', 'w') as f:
        for fpath, pred in zip(file_paths, predictions):
            f.write(f"File: {fpath} - Prediction: {pred}\n")


if __name__ == '__main__':
    predict_with_pretrained_model()
