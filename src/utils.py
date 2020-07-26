import os
import random
import torch
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_cuda_device(cuda_num):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_num)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    torch.backends.cudnn.deterministic = True


def write2log(text, logpath, epoch):
    add2log = ''
    if epoch == 0:
        add2log = '\t'.join(('epoch', 'train_loss', 'val_score', 'time'))
    add2log += f'\n{text}'
    with open(logpath, 'a') as log:
        log.write(add2log)
