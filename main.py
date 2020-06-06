#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-09-27 15:09:03
# @Author  : Jiabo (Raymond) Huang (jiabo.huang@qmul.ac.uk)
# @Link    : https://github.com/Raymond-sci

import torch
import torch.backends.cudnn as cudnn

import sys
import os
import time
from datetime import datetime

import models
import random


from lib import protocols
from lib.non_parametric_classifier import NonParametricClassifier
from lib.criterion import Criterion
from lib.ans_discovery import ANsDiscovery
from lib.utils import AverageMeter, time_progress, adjust_learning_rate

#from packages import session
from packages import lr_policy
from packages import optimizers
#from packages.config import CONFIG as cfg
from packages.loggers.std_logger import STDLogger as logger
from packages.loggers.tf_logger import TFLogger as SummaryWriter

import torchvision
import torchvision.transforms as transforms

import yaml

f=open('./configs/base.yaml')
d1=yaml.load(f)

f=open('./configs/cifar10.yaml')
d2=yaml.load(f)

for e in d2:
    d1[e]=d2[e]
cfg0=d1

class A():
    pass
cfg=A()
for e in cfg0:
    if not e:
        continue

    if type(cfg0[e])==str:
        exec('cfg.'+e+'=\''+str(cfg0[e])+'\'')
    else:
        exec('cfg.'+e+'='+str(cfg0[e]))

batch=8


def require_args():
    
    # dataset to be used
    cfg.add_argument('--dataset', default='cifar10', type=str,
                        help='dataset to be used. (default: cifar10)')
    
    # network to be used
    cfg.add_argument('--network', default='resnet18', type=str,
                        help='backbone to be used. (default: ResNet18)')

    # optimizer to be used
    cfg.add_argument('--optimizer', default='sgd', type=str,
                        help='optimizer to be used. (default: sgd)')

    # lr policy to be used
    cfg.add_argument('--lr-policy', default='step', type=str,
                        help='lr policy to be used. (default: step)')

    # args for protocol
    cfg.add_argument('--protocol', default='knn', type=str,
                        help='protocol used to validate model')

    # args for network training
    cfg.add_argument('--max-epoch', default=200, type=int,
                        help='max epoch per round. (default: 200)')
    cfg.add_argument('--max-round', default=5, type=int, 
                        help='max iteration, including initialisation one. '
                             '(default: 5)')
    cfg.add_argument('--iter-size', default=1, type=int,
                        help='caffe style iter size. (default: 1)')
    cfg.add_argument('--display-freq', default=1, type=int,
                        help='display step')
    cfg.add_argument('--test-only', action='store_true', 
                        help='test only')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
def main():




    best_acc = 0.  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    start_round = 0 # start for iter 0 or last checkpoint iter
    
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                          shuffle=False, num_workers=2)
 
    #trainset, trainloader, testset, testloader = datasets.get(cfg.dataset, instant=True)
    # cheat labels are used to compute neighbourhoods consistency only
    cheat_labels = torch.tensor(trainset.targets).long().to(device)
    ntrain, ntest = len(trainset), len(testset)

    net = models.get(cfg.network, instant=True)
    npc = NonParametricClassifier(128, ntrain, cfg.npc_temperature, cfg.npc_momentum)
    ANs_discovery = ANsDiscovery(ntrain)
    criterion = Criterion()
    optimizer = optimizers.get(cfg.optimizer, instant=True, params=net.parameters())
    lr_handler = lr_policy.get(cfg.lr_policy, instant=True)
    protocol = protocols.get(cfg.protocol)
    
    # data parallel
    """
    if device == 'cuda':
        if (cfg.network.lower().startswith('alexnet') or
            cfg.network.lower().startswith('vgg')):
            net.features = torch.nn.DataParallel(net.features,
                                    device_ids=range(len(cfg.gpus.split(','))))
        else:
            net = torch.nn.DataParallel(net, device_ids=range())
        cudnn.benchmark = True
    """

    net, npc, ANs_discovery, criterion = (net.to(device), npc.to(device), 
        ANs_discovery.to(device), criterion.to(device))
    
    # load ckpt file if necessary

    if cfg.resume:
        assert os.path.exists(cfg.resume), "Resume file not found: %s" % cfg.resume
        ckpt = torch.load(cfg.resume)
        net.load_state_dict(ckpt['net'])
        optimizer.load_state_dict(ckpt['optimizer'])
        npc = npc.load_state_dict(ckpt['npc'])
        ANs_discovery.load_state_dict(ckpt['ANs_discovery'])
        best_acc = ckpt['acc']
        start_epoch = ckpt['epoch']
        start_round = ckpt['round']
    
    # test if necessary
    if cfg.test_only:
        print('Testing at beginning...')
        acc = protocol(net, npc, trainloader, testloader, 200,
                            cfg.npc_temperature, True, device)
        print('Evaluation accuracy at %d round and %d epoch: %.2f%%' %
                                        (start_round, start_epoch, acc * 100))
        sys.exit(0)

    print('Start the progressive training process from round: %d, '
        'epoch: %d, best acc is %.4f...' % (start_round, start_epoch, best_acc))
    round = start_round
  
    while (round < cfg.max_round):

        # variables are initialized to different value in the first round
        is_first_round = True if round == start_round else False
        #best_acc = best_acc if is_first_round else 0
        print(round, start_round, best_acc)
    
        if not is_first_round:
            ANs_discovery.update(round, npc, cheat_labels)
            print('ANs consistency at %d round is %.2f%%' %
                        (round, ANs_discovery.consistency * 100))

        ANs_num = ANs_discovery.anchor_indexes.shape[0]
    

        # declare local writer
  


        # start to train for an epoch
        epoch = start_epoch if is_first_round else 0
        lr = cfg.base_lr
        while lr > 0 and epoch < cfg.max_epoch:
           
            # get learning rate according to current epoch
            lr = lr_handler.update(epoch)

            train(round, epoch, net, trainloader, optimizer, npc, criterion,
                ANs_discovery, lr)


            acc = protocol(net, npc, trainloader, testloader, 200,
                            cfg.npc_temperature, False, device)
       
            #logger.info('Evaluation accuracy at %d round and %d epoch: %.1f%%'
            #                      % (round, epoch, acc * 100))
            #logger.info('Best accuracy at %d round and %d epoch: %.1f%%'
            #                               % (round, epoch, best_acc * 100))

            is_best = acc >= best_acc
            best_acc = max(acc, best_acc)
            print(acc)
            if is_best:
                print('*****************')
                target = os.path.join('./tmp/', '%04d-%05d.ckpt'
                                                        % (round, ANs_num))
                #logger.info(str(epoch)+ (' :Saving checkpoint: %s' % str(acc)))
    
                state = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'ANs_discovery' : ANs_discovery.state_dict(),
                    'npc' : npc.state_dict(),
                    'acc': acc,
                    'epoch': epoch + 1,
                    'round' : round
                }
                torch.save(state, target)
            epoch += 1

        # log best accuracy after each iteration
        print(best_acc,'***********')
        round += 1

# Training
def train(round, epoch, net, trainloader, optimizer, npc, criterion,
            ANs_discovery, lr):
    #torch.cuda.empty_cache()

    # tracking variables
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch the model to train mode
    net.train()
    # adjust learning rate
    adjust_learning_rate(optimizer, lr)

    end = time.time()
    start_time = datetime.now()
    optimizer.zero_grad()
    a = []
    for batch_idx, (inputs,targets) in enumerate(trainloader):
        a.append([batch_idx,inputs,targets])
    random.shuffle(a)
    for e in a:
        batch_idx, inputs, targets = e
        data_time.update(time.time() - end)
        indexes = torch.tensor([k for k in range(batch_idx*batch,(batch_idx+1)*batch)])
     
        inputs, indexes = inputs.to(device), indexes.to(device)

        features = net(inputs)
        outputs = npc(features, indexes)
        loss = criterion(outputs, indexes, ANs_discovery, round, 5) / cfg.iter_size

        loss.backward()
        train_loss.update(loss.item() * cfg.iter_size, inputs.size(0))

        if batch_idx % cfg.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % cfg.display_freq != 0:
            continue



        elapsed_time, estimated_time = time_progress(batch_idx + 1,
                                        len(trainloader), batch_time.sum)
        """
        print('Round: {round} Epoch: {epoch}/{tot_epochs} '
              'Progress: {elps_iters}/{tot_iters} ({elps_time}/{est_time}) '
              'Data: {data_time.avg:.3f} LR: {learning_rate:.5f} '
              'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f})'.format(
              round=round, epoch=epoch, tot_epochs=cfg.max_epoch,
              elps_iters=batch_idx, tot_iters=len(trainloader),
              elps_time=elapsed_time, est_time=estimated_time,
              data_time=data_time, learning_rate=lr,
              train_loss=train_loss))
        """

if __name__ == '__main__':
    
    main()
