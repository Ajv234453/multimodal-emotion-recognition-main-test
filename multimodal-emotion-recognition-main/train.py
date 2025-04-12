'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_accuracy

def train_epoch_multimodal(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    model.train()

    batch_time = AverageMeter()#记录，监控损失和准确性指标
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
        
    end_time = time.time()
    for i, (audio_inputs, visual_inputs,road_inputs,targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        # print(road_inputs.shape)
        # print(audio_inputs.shape)
        # print(visual_inputs.shape)
        targets = Variable(targets)
        targets = targets.to(opt.device)
            
        if opt.mask is not None:
            with torch.no_grad():
                if opt.mask == 'noise':
                    audio_inputs = torch.cat((audio_inputs, torch.randn(audio_inputs.size()), audio_inputs), dim=0)                   
                    visual_inputs = torch.cat((visual_inputs, visual_inputs, torch.randn(visual_inputs.size())), dim=0)
                    targets = torch.cat((targets, targets, targets), dim=0)                    
                    shuffle = torch.randperm(audio_inputs.size()[0])
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    targets = targets[shuffle]
                    
                elif opt.mask == 'softhard':
                    # 生成音频和路况输入的系数
                    coefficients = torch.randint(low=0, high=100, size=(audio_inputs.size(0), 1, 1)) / 100
                    road_coefficients = torch.randint(low=0, high=100, size=(road_inputs.size(0), 1, 1)) / 100

                    # 计算视觉输入的系数（1减去音频输入的系数）
                    vision_coefficients = 1 - coefficients

                    # 扩展系数以匹配各自输入的尺寸
                    coefficients = coefficients.repeat(1, audio_inputs.size(1), audio_inputs.size(2))
                    vision_coefficients = vision_coefficients.unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                                                                 visual_inputs.size(1),
                                                                                                 visual_inputs.size(2),
                                                                                                 visual_inputs.size(3),
                                                                                                 visual_inputs.size(4))
                    # 假设 road_inputs 的形状为 [N, C, D, H, W]
                    road_coefficients = torch.randint(low=0, high=100,
                                                      size=(road_inputs.size(0), 1, 1, 1, 1)).float() / 100
                    # 扩展系数以匹配 road_inputs 的形状，除了通道维度
                    road_coefficients = road_coefficients.repeat(1, road_inputs.size(1), 1, 1, 1)

                    # 构建修改后的输入
                    audio_inputs = torch.cat(
                        (audio_inputs, audio_inputs * coefficients, torch.zeros(audio_inputs.size()), audio_inputs),
                        dim=0)
                    road_inputs = torch.cat((road_inputs, road_inputs * road_coefficients, road_inputs,
                                             torch.zeros(road_inputs.size())),dim=0)
                    visual_inputs = torch.cat((visual_inputs, visual_inputs * vision_coefficients, visual_inputs,
                                               torch.zeros(visual_inputs.size())), dim=0)

                    # 扩展目标以匹配修改后的输入尺寸
                    targets = torch.cat((targets, targets, targets, targets), dim=0)

                    # 生成打乱顺序的随机排列
                    shuffle = torch.randperm(audio_inputs.size()[0])

                    # 使用随机排列打乱所有输入和目标的顺序
                    audio_inputs = audio_inputs[shuffle]
                    visual_inputs = visual_inputs[shuffle]
                    road_inputs = road_inputs[shuffle]
                    targets = targets[shuffle]

        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        road_inputs = road_inputs.permute(0, 2, 1, 3, 4)
        road_inputs = road_inputs.reshape(road_inputs.shape[0] * road_inputs.shape[1], road_inputs.shape[2],road_inputs.shape[3], road_inputs.shape[4])
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)
        road_inputs = Variable(road_inputs)
        # print(road_inputs.shape)
        # print(audio_inputs.shape)
        # print(visual_inputs.shape)
        targets = Variable(targets)
        outputs = model(audio_inputs, visual_inputs,road_inputs)
        loss = criterion(outputs, targets)

        losses.update(loss.data, audio_inputs.size(0))
        prec1, prec5 = calculate_accuracy(outputs.data, targets.data, topk=(1,5))
        top1.update(prec1, audio_inputs.size(0))
        top5.update(prec5, audio_inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val.item(),
            'prec1': top1.val.item(),
            'prec5': top5.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.5f} ({top1.avg:.5f})\t'
                  'Prec@5 {top5.val:.5f} ({top5.avg:.5f})'.format(
                      epoch,
                      i,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                      lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg.item(),
        'prec1': top1.avg.item(),
        'prec5': top5.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })

 
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger):
    print('train at epoch {}'.format(epoch))
    
    if opt.model == 'multimodalcnn':
        train_epoch_multimodal(epoch,  data_loader, model, criterion, optimizer, opt, epoch_logger, batch_logger)
        return
    
    
