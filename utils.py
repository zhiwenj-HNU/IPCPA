# 模板序号008
# 开发时间 2022/11/20 11:18
""" helper function

author baiyu
"""
import os
import sys
import re
import datetime

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_network(args):
    """ return given network
    """

    if   args.net == 'ECA_ResNet_50':
        from models.ECA_ResNet import ECA_ResNet_50
        net = ECA_ResNet_50()
    elif args.net == 'ECA_ResNet_101':
        from models.ECA_ResNet import ECA_ResNet_101
        net = ECA_ResNet_101()
    elif args.net == 'ECA_ResNet_152':
        from models.ECA_ResNet import ECA_ResNet_152
        net = ECA_ResNet_152()




    elif args.net == 'Inception_ResNetv2':
        from models.Inception_ResNetv2 import Inception_ResNetv2
        net = Inception_ResNetv2()


    elif args.net == 'IPC_ResNet_18':
        from models.IPC_ResNet import IPC_ResNet_18
        net = IPC_ResNet_18()
    elif args.net == 'IPC_ResNet_34':
        from models.IPC_ResNet import IPC_ResNet_34
        net = IPC_ResNet_34()
    elif args.net == 'IPC_ResNet_50':
        from models.IPC_ResNet import IPC_ResNet_50
        net = IPC_ResNet_50()
    elif args.net == 'IPC_ResNet_101':
        from models.IPC_ResNet import IPC_ResNet_101
        net = IPC_ResNet_101()


    elif args.net == 'IPCPA_ResNet_18':
        from models.IPCPA_ResNet import IPCPA_ResNet_18
        net = IPCPA_ResNet_18()
    elif args.net == 'IPCPA_ResNet_34':
        from models.IPCPA_ResNet import IPCPA_ResNet_34
        net = IPCPA_ResNet_34()
    elif args.net == 'IPCPA_ResNet_50':
        from models.IPCPA_ResNet import IPCPA_ResNet_50
        net = IPCPA_ResNet_50()
    elif args.net == 'IPCPA_ResNet_101':
        from models.IPCPA_ResNet import IPCPA_ResNet_101
        net = IPCPA_ResNet_101()


    elif args.net == 'IPP_ResNet_18':
        from models.IPP_ResNet import IPP_ResNet_18
        net = IPP_ResNet_18()
    elif args.net == 'IPP_ResNet_34':
        from models.IPP_ResNet import IPP_ResNet_34
        net = IPP_ResNet_34()
    elif args.net == 'IPP_ResNet_50':
        from models.IPP_ResNet import IPP_ResNet_50
        net = IPP_ResNet_50()
    elif args.net == 'IPP_ResNet_101':
        from models.IPP_ResNet import IPP_ResNet_101
        net = IPP_ResNet_101()


    elif args.net == 'IPM1_ResNet_18':
        from models.IPM1_ResNet import IPM1_ResNet_18
        net = IPM1_ResNet_18()
    elif args.net == 'IPM1_ResNet_34':
        from models.IPM1_ResNet import IPM1_ResNet_34
        net = IPM1_ResNet_34()
    elif args.net == 'IPM1_ResNet_50':
        from models.IPM1_ResNet import IPM1_ResNet_50
        net = IPM1_ResNet_50()
    elif args.net == 'IPM1_ResNet_101':
        from models.IPM1_ResNet import IPM1_ResNet_101
        net = IPM1_ResNet_101()


    elif args.net == 'IPM2_ResNet_18':
        from models.IPM2_ResNet import IPM2_ResNet_18
        net = IPM2_ResNet_18()
    elif args.net == 'IPM2_ResNet_34':
        from models.IPM2_ResNet import IPM2_ResNet_34
        net = IPM2_ResNet_34()
    elif args.net == 'IPM2_ResNet_50':
        from models.IPM2_ResNet import IPM2_ResNet_50
        net = IPM2_ResNet_50()
    elif args.net == 'IPM2_ResNet_101':
        from models.IPM2_ResNet import IPM2_ResNet_101
        net = IPM2_ResNet_101()


    elif args.net == 'ResNet_18':
        from models.ResNet import ResNet_18
        net = ResNet_18()
    elif args.net == 'ResNet_34':
        from models.ResNet import ResNet_34
        net = ResNet_34()
    elif args.net == 'ResNet_50':
        from models.ResNet import ResNet_50
        net = ResNet_50()
    elif args.net == 'ResNet_101':
        from models.ResNet import ResNet_101
        net = ResNet_101()


    elif args.net == 'ResNeXt_50':
        from models.ResNeXt import ResNeXt_50
        net = ResNeXt_50()
    elif args.net == 'ResNeXt_101':
        from models.ResNeXt import ResNeXt_101
        net = ResNeXt_101()

    elif args.net == 'SE_ResNet_18':
        from models.SE_ResNet import SE_ResNet_18
        net = SE_ResNet_18()
    elif args.net == 'SE_ResNet_34':
        from models.SE_ResNet import SE_ResNet_34
        net = SE_ResNet_34()
    elif args.net == 'SE_ResNet_50':
        from models.SE_ResNet import SE_ResNet_50
        net = SE_ResNet_50()
    elif args.net == 'SE_ResNet_101':
        from models.SE_ResNet import SE_ResNet_101
        net = SE_ResNet_101()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu:                                         # CUDA
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform = transforms.Compose([transforms.Resize((64, 64)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(15),
                                    transforms.ToTensor(),
                                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],       # Normalize Optional
                                    #                      std=[0.29, 0.224, 0.225])
                                     transforms.Normalize(mean, std)])
    Face_training = ImageFolder(root=r'E:\IPCPA\data\train', transform=transform)              # train data storage path, modified as needed
    Face_training_loader=DataLoader(Face_training,batch_size=batch_size,shuffle=True,num_workers=num_workers)

    # transform_train = transforms.Compose([
    #     #transforms.ToPILImage(),
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(15),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean, std)
    # ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    # cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    # cifar100_training_loader = DataLoader(
    #     cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return Face_training_loader    ##########################################################

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):        # data load

    transform_test = transforms.Compose([
        transforms.Resize((64, 64)),                           # Resize the picture
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],     # Normalize Optional
                             #std=[0.29, 0.224, 0.225])
        transforms.Normalize(mean, std)
    ])
    Face_val_data = ImageFolder(root=r'E:\IPCPA\data\test', transform=transform_test)      # test data storage path, modified as needed
    Face_val_loader = DataLoader(Face_val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    # cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    # cifar100_test_loader = DataLoader(
    #     cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    #Face_val_loader = ImageFolder(root=r'E:\pycharm_project_pytorch\Face_emotion\genki4k\face\test',
                                  #transform=transform_test)

    return Face_val_loader

def compute_mean_std(cifar100_dataset):
    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    folders = os.listdir(net_weights)
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''
    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):

    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]