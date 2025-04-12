from datasets.ravdess import RAVDESS

def get_training_set(opt, spatial_transform=None, audio_transform=None,road_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
    #确保opt.dataset的值是'RAVDESS'。如果不是，它将打印一个错误消息并终止程序执行
    if opt.dataset == 'RAVDESS':
        training_data = RAVDESS(
            opt.annotation_path,
            'training',
            spatial_transform=spatial_transform, data_type='audiovisualroad', audio_transform=audio_transform)
    return training_data


def get_validation_set(opt, spatial_transform=None, audio_transform=None,road_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'RAVDESS':
        validation_data = RAVDESS(
            opt.annotation_path,
            'validation',
            spatial_transform=spatial_transform, data_type = 'audiovisualroad', audio_transform=audio_transform)
    return validation_data


def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['RAVDESS'], print('Unsupported dataset: {}'.format(opt.dataset))
    assert opt.test_subset in ['val', 'test']

    if opt.test_subset == 'val':
        subset = 'validation'
    elif opt.test_subset == 'test':
        subset = 'testing'
    if opt.dataset == 'RAVDESS':
        test_data = RAVDESS(
            opt.annotation_path,
            subset,
            spatial_transform=spatial_transform, data_type='audiovisualroad',audio_transform=audio_transform)
    return test_data
