#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-27 09:38:53
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import os
import time

import torch
import importlib
import numpy as np



from config import CONFIG as cfg


def require_args():

    # timestamp
    stt = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
    tt = int(time.time())

    cfg.add_argument('--session', default=stt, type=str,
                        help='session name (default: %s)' % stt)
    cfg.add_argument('--sess-dir', default='sessions', type=str,
                        help='directory to store session. (default: sessions)')
    cfg.add_argument('--print-args', action='store_true',
                        help='do nothing but print all args. (default: False)')
    cfg.add_argument('--seed', default=tt, type=int,
                        help='session random seed. (default: %d)' % tt)
    cfg.add_argument('--brief', action='store_true',
                        help='print log with priority higher than debug. '
                             '(default: False)')
    cfg.add_argument('--debug', action='store_true',
                        help='if debugging, no log or checkpoint files will be stored. '
                        '(default: False)')
    cfg.add_argument('--gpus', default='', type=str,
                        help='available gpu list. (default: \'\')')
    cfg.add_argument('--resume', default=None, type=str,
                        help='path to resume session. (default: None)')
    cfg.add_argument('--restart', action='store_true',
                        help='load session status and start a new one. '
                             '(default: False)')

def run(main):

    # import main module
    main = importlib.import_module(main)
    # parse args
    main.require_args()
    cfg.parse()

    # setup session according to args
    setup()

    # run main function
    main.main()

def setup():
    """
    set up common environment for training
    """

    # print args
    if not cfg.brief:
        print(cfg)
    # exit if require to print args only
    if cfg.print_args:
        exit(0)

 
    # fix random seeds
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    
    # set visible gpu devices at main function
  
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus

    # setup session name and session path
    if cfg.resume and cfg.resume.strip() != '' and not cfg.restart:
        assert os.path.exists(cfg.resume), ("Resume file not "
                                                "found: %s" % cfg.resume)
        ckpt = torch.load(cfg.resume)
        if 'session' in ckpt:
            cfg.session = ckpt['session']
    cfg.sess_dir = os.path.join(cfg.sess_dir, cfg.session)
    

    # setup checkpoint dir
    cfg.ckpt_dir = os.path.join(cfg.sess_dir, 'checkpoint')
    if not os.path.exists(cfg.ckpt_dir) and not cfg.debug:
        os.makedirs(cfg.ckpt_dir)



