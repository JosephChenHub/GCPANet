#!/usr/bin/python3
#coding=utf-8

import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from data import dataset
from net  import GCPANet
import logging as logger
from lib.data_prefetcher import DataPrefetcher
from lib.lr_finder import LRFinder
import numpy as np
import matplotlib.pyplot as plt

TAG = "ours"
SAVE_PATH = "ours"
logger.basicConfig(level=logger.INFO, format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S', \
                           filename="train_%s.log"%(TAG), filemode="w")


""" set lr """
def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum

BASE_LR = 1e-3
MAX_LR = 0.1
FIND_LR = False #True


def train(Dataset, Network):
    ## dataset
    cfg    = Dataset.Config(datapath='./data/DUTS', savepath=SAVE_PATH, mode='train', batch=8, lr=0.05, momen=0.9, decay=5e-4, epoch=30)
    data   = Dataset.Data(cfg)
    loader = DataLoader(data, batch_size=cfg.batch, shuffle=True, num_workers=8)
    prefetcher = DataPrefetcher(loader)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer   = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    sw          = SummaryWriter(cfg.savepath)
    global_step = 0

    db_size = len(loader)
    if FIND_LR:
        lr_finder = LRFinder(net, optimizer, criterion=None)
        lr_finder.range_test(loader, end_lr=50, num_iter=100, step_mode="exp")
        plt.ion()
        lr_finder.plot()
        import pdb; pdb.set_trace()

    #training
    for epoch in range(cfg.epoch):
        prefetcher = DataPrefetcher(loader)
        batch_idx = -1
        image, mask = prefetcher.next()
        while image is not None:
            niter = epoch * db_size + batch_idx
            lr, momentum = get_triangle_lr(BASE_LR, MAX_LR, cfg.epoch*db_size, niter, ratio=1.)
            optimizer.param_groups[0]['lr'] = 0.1 * lr #for backbone
            optimizer.param_groups[1]['lr'] = lr
            optimizer.momentum = momentum
            batch_idx += 1
            global_step += 1
            out2, out3, out4, out5 = net(image)
            loss2                  = F.binary_cross_entropy_with_logits(out2, mask)
            loss3                  = F.binary_cross_entropy_with_logits(out3, mask)
            loss4                  = F.binary_cross_entropy_with_logits(out4, mask)
            loss5                  = F.binary_cross_entropy_with_logits(out5, mask)
            loss                   = loss2*1 + loss3*0.8 + loss4*0.6 + loss5*0.4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            sw.add_scalars('loss', {'loss2':loss2.item(), 'loss3':loss3.item(), 'loss4':loss4.item(), 'loss5':loss5.item(), 'loss':loss.item()}, global_step=global_step)
            if batch_idx % 10 == 0:
                msg = '%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | loss2=%.6f | loss3=%.6f | loss4=%.6f | loss5=%.6f'%(datetime.datetime.now(),  global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item())
                print(msg)
                logger.info(msg)
            image, mask = prefetcher.next()

        if (epoch+1)%10 == 0 or (epoch+1)==cfg.epoch:
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))



if __name__=='__main__':
    train(dataset, GCPANet)
