import os
import sys
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from apex import amp

import config
from cosine import CosineAnnealingWarmUpRestarts
from dataset import get_train_valid_datasets
from metric import alaska_weighted_auc
from model import get_model
from utils import set_seed, set_cuda_device, write2log


def train_epoch(loader, model, criterion, optimizer, device, scheduler=None, apex=False):
    model.train()
    sum_loss = 0.
    optimizer.zero_grad()
    pbar = tqdm(enumerate(loader), total=len(loader))
    for i, (imgs, labels) in pbar:
        labels = torch.argmax(labels, dim=1)
        imgs, labels = imgs.to(device), labels.to(device).long()
        loss = criterion(model(imgs), labels)
        if apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        sum_loss += loss.item()
        pbar.set_description(f'Loss: {sum_loss / (i + 1)}')
        if scheduler:
            scheduler.step()
    return sum_loss / len(loader)


def test_epoch(loader, model, device):
    preds = []
    targets = []
    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(device), labels.cpu().numpy()
            pred = 1 - F.softmax(model(imgs), dim=1).cpu().numpy()[:, 0]
            preds.extend(pred)
            targets.extend(labels.argmax(axis=1).clip(min=0, max=1).astype(int))
    return alaska_weighted_auc(targets, preds)


def train(model, train_loader, valid_loader, config):
    if config.log_wandb:
        wandb.init(project=f'{config.workdir}')
    os.makedirs(f'model/{config.workdir}', exist_ok=True)
    model.to(config.device)
    criterion = getattr(nn, config.loss)()
    optimizer = getattr(torch.optim, config.optimizer['name'])(model.parameters(), lr=config.optimizer['params']['lr'])
    if config.apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    n_iter_per_ep = len(train_loader)
    scheduler = getattr(sys.modules[__name__], config.scheduler['name'])(
        optimizer,
        T_0=config.scheduler['params']['T_0']*n_iter_per_ep,
        T_mult=config.scheduler['params']['T_mult'],
        eta_max=config.scheduler['params']['eta_max'],
        T_up=config.scheduler['params']['T_up']*n_iter_per_ep,
    )

    max_score = 0

    for epoch in range(config.n_epochs):
        start_time = time.time()
        train_loss = train_epoch(train_loader, model, criterion, optimizer, config.device, scheduler, config.apex)
        valid_score = test_epoch(valid_loader, model, config.device)
        elapsed_time = time.time() - start_time
#         scheduler.step()
        output = '\t'.join(map(str, (epoch, round(train_loss, 4), round(valid_score, 4), round(elapsed_time, 2))))
        print(output)
        write2log(output, f'model/{config.workdir}/{config.workdir}.txt', epoch)

        if config.log_wandb:
            wandb.log({'train_loss': train_loss, 'valid_score': valid_score})

        if max_score < valid_score:
            max_score = valid_score
            torch.save(model, f'model/{config.workdir}/best_model.pth')


if __name__ == '__main__':
    set_seed(config.seed)
    set_cuda_device(config.gpu)
    model = get_model(config.model_name, config.n_classes)
    train_dataset, valid_dataset = get_train_valid_datasets(config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    train(model, train_loader, valid_loader, config)
