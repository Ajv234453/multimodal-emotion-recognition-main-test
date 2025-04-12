# # -*- coding: utf-8 -*-
#
# import os
# root = 'D:/DL/Video_Speech_Actor_24'
# #splits used in the paper with 5 folds
# #n_folds=5
# #folds =[ [[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]],[[5,6,7,8],[9,10,11,12],[13,14,15,16,17,18,19,20,21,22,23,24,1,2,3,4]],[[9,10,11,12],[13,14,15,16],[17,18,9,20,21,22,23,24,1,2,3,4,5,6,7,8]],[[13,14,15,16],[17,18,19,20],[21,22,23,24,1,2,3,4,5,6,7,8,9,10,11,12]],[[17,18,19,20],[21,22,23,24],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]]
# #
# # n_folds=1
# # folds = [[[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]]
# n_folds=1
# folds = [[[0], [1], [2,3]]]
# for fold in range(n_folds):
#         fold_ids = folds[fold]
#         train_ids,test_ids,val_ids = fold_ids
#         print(test_ids)
#         print(val_ids)
#         print(train_ids)
#
#         #annotation_file = 'annotations_croppad_fold'+str(fold+1)+'.txt'
#         annotation_file = 'annotations.txt'
#
#         for i,actor in enumerate(os.listdir(root)):
#             for video in os.listdir(os.path.join(root, actor)):
#                 if not video.endswith('.npy') or 'croppad' not in video:
#                     continue
#                 label = str(int(video.split('_')[1]))
#
#                 audio = '0' + video.split('_face')[0][1:] + '_croppad.wav'
#                 if i in train_ids:
#                    with open(annotation_file, 'a') as f:
#                        f.write(os.path.join(root,actor, video) + ';' + os.path.join(root,actor, audio) + ';' + label + ';training' + '\n')
#
#
#                 elif i in val_ids:
#                     with open(annotation_file, 'a') as f:
#                         f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';validation' + '\n')
#
#                 else:
#                     with open(annotation_file, 'a') as f:
#                         f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ label + ';testing' + '\n')

import os

root = 'D:\CCFAcode\Video_Speech_Actor_24'
n_folds = 1
folds = [[[0,1,2,3,4], [5],[6,7]]]

for fold in range(n_folds):
    fold_ids = folds[fold]
    #train_ids, val_ids = fold_ids
    train_ids, val_ids,test_ids = fold_ids

    # 用于记录注释的文件
    annotation_file = 'annotations.txt'

    for i, actor in enumerate(os.listdir(root)):
        for audio in os.listdir(os.path.join(root, actor)):
            # 跳过不是.npy文件或不包含'croppad'的文件
            #if (not audio.endswith('.wav') and 'croppad' not in audio):
            if not (audio.endswith('.wav') and 'croppad' in audio):

                continue
            #print(audio)
            # 提取标签
            label = str(int(audio.split('_')[1]))

            # 构造音频文件路径
            video =audio.split('_croppad')[0][:] + '_facecroppad.npy'
            print(video)
            # 检查是否存在对应的road数据文件
            road = audio.split('_croppad')[0][:] + '_r_facecroppad.npy'
            # 根据actor索引将数据分配到训练、测试或验证集
            if i in train_ids:
                set_type = 'training'
            elif i in test_ids:
                set_type = 'testing'
            elif i in val_ids:
                set_type = 'validation'

            # 写入注释文件
            with open(annotation_file, 'a') as f:
                entry = f"{os.path.join(root, actor, video)};{os.path.join(root, actor, audio)};{os.path.join(root, actor, road)};{label};{set_type}"
                entry += '\n'
                f.write(entry)

