# 模板序号008
# 开发时间 2022/11/20 11:48
# config
import os
from datetime import datetime

#CIFAR100 dataset path (python version)
#CIFAR100_PATH = '/nfs/private/cifar100/cifar-100-python'

#mean and std of cifar100 dataset
CIFAR10_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR10_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

#CIFAR100_TEST_MEAN = (0.5088964127604166, 0.48739301317401956, 0.44194221124387256)
#CIFAR100_TEST_STD = (0.2682515741720801, 0.2573637364478126, 0.2770957707973042)

#directory to save weights file
CHECKPOINT_PATH = 'save models'
MILESTONES = [60, 120, 160]
DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'

EPOCH = 200                                                   #

TIME_NOW = datetime.now().strftime(DATE_FORMAT)
LOG_DIR = 'runing results'
SAVE_EPOCH = 20








