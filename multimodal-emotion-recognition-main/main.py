# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 14:07:29 2021

@author: chumache
"""

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

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    opt = parse_opts()
    n_folds = 1#交叉验证的折叠数
    test_accuracies = []#存储每个折叠的测试精度
    
    if opt.device != 'cpu':
        #这里先改成cpu
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #opt.device = 'cuda' if 2==3 else 'cpu'
        #根据opt.device的设置选择使用CPU还是GPU
    pretrained = opt.pretrain_path != 'None'
    #检查是否指定了预训练模型的路径

    #opt.result_path = 'res_'+str(time.time())
    if not os.path.exists(opt.result_path):
        os.makedirs(opt.result_path)
        #如果结果目录不存在，则创建一个
    opt.arch = '{}'.format(opt.model)
    #将opt.model的值格式化为一个字符串并赋给opt.arch，{}为占位符
    opt.store_name = '_'.join([opt.dataset, opt.model, str(opt.sample_duration)])
            
    for fold in range(n_folds):
        #if opt.dataset == 'RAVDESS':
        #    opt.annotation_path = '/lustre/scratch/chumache/ravdess-develop/annotations_croppad_fold'+str(fold+1)+'.txt'

        print(opt)
        with open(os.path.join(opt.result_path, 'opts'+str(time.time())+str(fold)+'.json'), 'w') as opt_file:
            json.dump(vars(opt), opt_file)
        #opt对象的所有属性及其值保存位于opt.result_path指定的目录的文件中
        torch.manual_seed(opt.manual_seed)
        #设置随机种子
        model, parameters = generate_model(opt)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(opt.device)
        #创建一个交叉熵损失函数，并将其赋值给变量criterion
        #该损失函数迁移到指定的设备上，以确保后续的计算发生在正确的设备上
        if not opt.no_train:
        #根据opt.no_train值指示是否跳过训练过程
            #组合多个数据转换操作
            video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),#随机地对视频进行水平翻转
                transforms.RandomRotate(),#随机旋转视频
                transforms.ToTensor(opt.video_norm_value)])#将视频数据转换为张量
        
            training_data = get_training_set(opt, spatial_transform=None)
            #获取训练数据
            train_loader = torch.utils.data.DataLoader(
                training_data,
                #这里修改暂时修改成4
                batch_size=opt.batch_size,
                #batch_size=4,
                shuffle=True,
                num_workers=opt.n_threads,
                pin_memory=True,
                collate_fn = custom_collate_fn  # 指定自定义的 collate 函数
        )
            #utils.py文件中定义的日志记录器
            train_logger = Logger(
                os.path.join(opt.result_path, 'train'+str(fold)+'.log'),
                ['epoch', 'loss', 'prec1', 'prec5', 'lr'])
            train_batch_logger = Logger(
                os.path.join(opt.result_path, 'train_batch'+str(fold)+'.log'),
                ['epoch', 'batch', 'iter', 'loss', 'prec1', 'prec5', 'lr'])
            
            #优化器
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=opt.lr_patience)
        #模型验证的初始化
        if not opt.no_val:
            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                #将视频帧转换为张量格式，并使用指定的video_norm_value对其进行归一化
            validation_data = get_validation_set(opt, spatial_transform=None)
            #
            val_loader = torch.utils.data.DataLoader(
                validation_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True,
                collate_fn=custom_collate_fn  # 指定自定义的 collate 函数
            )
        
            val_logger = Logger(
                    os.path.join(opt.result_path, 'val'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])
            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            
        best_prec1 = 0
        best_loss = 1e10
        if opt.resume_path:
            print('loading checkpoint {}'.format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path)
            #加载检查点文件。这个文件通常包含模型的参数、优化器的状态、以及其他相关信息
            assert opt.arch == checkpoint['arch']
            #确认当前模型的架构（opt.arch）与检查点中保存的架构（checkpoint['arch']）相匹配
            best_prec1 = checkpoint['best_prec1']
            opt.begin_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])

        for i in range(opt.begin_epoch, opt.n_epochs + 1):

            if not opt.no_train:
                adjust_learning_rate(optimizer, i, opt)
                train_epoch(i, train_loader, model, criterion, optimizer, opt,
                            train_logger, train_batch_logger)
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1
                    }
                save_checkpoint(state, False, opt, fold)
            
            if not opt.no_val:
                validation_loss, prec1 = val_epoch(i, val_loader, model, criterion, opt,
                                            val_logger)
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                state = {
                'epoch': i,
                'arch': opt.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_prec1': best_prec1
                }
               
                save_checkpoint(state, is_best, opt, fold)

               
        if opt.test:

            test_logger = Logger(
                    os.path.join(opt.result_path, 'test'+str(fold)+'.log'), ['epoch', 'loss', 'prec1', 'prec5'])

            video_transform = transforms.Compose([
                transforms.ToTensor(opt.video_norm_value)])
                
            test_data = get_test_set(opt, spatial_transform=video_transform) 
        
            #load best model
            best_state = torch.load('%s/%s_best' % (opt.result_path, opt.store_name)+str(fold)+'.pth')
            model.load_state_dict(best_state['state_dict'])
        
            test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=opt.batch_size,
                shuffle=False,
                num_workers=opt.n_threads,
                pin_memory=True,
                collate_fn=custom_collate_fn  # 指定自定义的 collate 函数
            )
            
            test_loss, test_prec1 = val_epoch(10000, test_loader, model, criterion, opt,
                                            test_logger)
            
            with open(os.path.join(opt.result_path, 'test_set_bestval'+str(fold)+'.txt'), 'a') as f:
                    f.write('Prec1: ' + str(test_prec1) + '; Loss: ' + str(test_loss))
            test_accuracies.append(test_prec1) 
                
            
    with open(os.path.join(opt.result_path, 'test_set_bestval.txt'), 'a') as f:
        f.write('Prec1: ' + str(np.mean(np.array(test_accuracies))) +'+'+str(np.std(np.array(test_accuracies))) + '\n')
