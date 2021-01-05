import argparse
import torch
from models import model_factory
from models.configs.detault import CONFIGS as config
from datasets.dataset_reader import create_dataloader
from solver.Solver import Solver
from preprocess.data_preprocess import TrainAugmentation, TestTransform
from models.rtm3d_loss import RTM3DLoss
from utils import check_point
from utils import utils
from utils import torch_utils
import os
import numpy as np
import tqdm
import cv2
from torch.utils.tensorboard import SummaryWriter
from fvcore.common.config import CfgNode
from utils.ParamList import ParamList

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = "model_{:07d}"
model_best = "model_best"


def setup(args):
    utils.init_seeds(20)
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    opt = CfgNode(args.__dict__)
    cfg.merge_from_other_cfg(opt)
    device = torch.device(cfg.DEVICE) if torch.cuda.is_available() else torch.device('cpu')
    cfg.update({'DEVICE': device})
    model = model_factory.create_model(cfg)
    dataloader = create_dataloader(cfg.DATASET.PATH, cfg,
                                   TrainAugmentation(cfg.INPUT_SIZE[0], mean=cfg.DATASET.MEAN),
                                   is_training=True)[0]

    test_dataloader = create_dataloader(cfg.DATASET.PATH, cfg,
                                   TestTransform(cfg.INPUT_SIZE[0], mean=cfg.DATASET.MEAN),
                                   is_training=True, split='test')[0]
    model.train()
    model.to(cfg.DEVICE)
    nb = len(dataloader)
    max_step_burn_in = max(3 * nb, 1e3)  # burn-in iterations, max(3 epochs, 1k iterations)
    # solver = Solver(model, cfg, max_steps_burn_in=max_step_burn_in, apex=amp)
    solver = Solver(model, cfg)
    rtm3d_loss = RTM3DLoss(cfg)
    return model, (dataloader, test_dataloader), solver, rtm3d_loss, cfg


def test_epoch(model, dataloader, rtm3d_loss, cfg):
    nb = len(dataloader)
    model.eval()
    pbar = tqdm.tqdm(enumerate(dataloader), total=nb)  # progress bar
    mloss = torch.tensor(0, dtype=torch.float32, device=cfg.DEVICE)
    num = 0
    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        with torch.no_grad():
            imgs = imgs.to(cfg.DEVICE)
            targets = targets.to(cfg.DEVICE)
            params = ParamList(targets.size)
            img_ids = targets.get_field('img_id')
            Ks = targets.get_field('K')
            Bs = imgs.shape[0]
            NKs = [None] * Bs
            for i in range(Bs):
                NKs[i] = Ks[img_ids == i][0:1, :]
            NKs = torch.cat(NKs, dim=0)
            pred = model(imgs, Ks=NKs)[1]
            loss, loss_items = rtm3d_loss(pred, targets)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return
            batch_size = imgs.shape[0]
            mloss += (loss * batch_size)
            num += batch_size

    print('test loss: %s' % (mloss / num))
    return mloss / num

import time


def train_epoch(model, dataloader, solver, rtm3d_loss, cfg, tb_writer, epoch):
    print(('\n' + '%10s' * 10) % ('Epoch', 'gpu_mem', 'MKF', 'M_OFF', 'DIM', 'DEPTH', 'ORIENT', 'total', 'targets', 'lr'))
    nb = len(dataloader)
    model.train()
    pbar = tqdm.tqdm(enumerate(dataloader), total=nb)  # progress bar
    mloss = torch.zeros((6,), dtype=torch.float32, device=cfg.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCH
    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        imgs = imgs.to(cfg.DEVICE)
        targets = targets.to(cfg.DEVICE)
        pred = model(imgs)
        loss, loss_items = rtm3d_loss(pred, targets)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return

        if i:
            mloss = (mloss + loss_items) / 2
        else:
            mloss = loss_items

        solver.step(loss)  ########

        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        mask = targets.get_field('mask')
        s = ('%10s' * 2 + '%10.4g' * 8) % (
            '%g/%g' % (epoch, epochs - 1), mem, *mloss, mask.shape[0], solver.learn_rate)
        pbar.set_description(s)

        # write tensorboard
        Tags = ['MKF', 'M_OFF', 'DIM', 'DEPTH', 'ORIENT', 'total']
        for x, tag in zip(list(mloss), Tags):
            tb_writer.add_scalar('loss/' + tag, x, epoch * nb + i)


def train(model, dataloader, solver, rtm3d_loss, cfg):
    tb_writer = SummaryWriter(comment="RTM3D Training")
    save_dir = os.path.join(cfg.TRAINING.WEIGHTS, cfg.MODEL.BACKBONE)
    logdir = os.path.join(cfg.TRAINING.LOGDIR, cfg.MODEL.BACKBONE)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    checkpointer = check_point.CheckPointer(model, solver,
                                            save_dir=save_dir,
                                            save_to_disk=True,
                                            mode='state-dict',
                                            device=cfg.DEVICE)
    start_epoch = 0
    min_loss = 10000
    arguments = {'epoch': start_epoch,
                 'min_loss': min_loss}
    if len(cfg.TRAINING.CHECKPOINT_FILE) > 0:
        ckpt = checkpointer.load(cfg.TRAINING.CHECKPOINT_FILE,
                                 use_latest=(cfg.TRAINING.CHECKPOINT_MODE != 'pretrained'),
                                 load_solver=cfg.SOLVER.LOAD_SOLVER)
        if 'epoch' in ckpt and cfg.TRAINING.CHECKPOINT_MODE == 'resume':
            start_epoch = ckpt['epoch'] + 1
        if 'min_loss' in ckpt and cfg.TRAINING.CHECKPOINT_MODE == 'resume':
            min_loss = ckpt['min_loss']
        del ckpt
    epochs = cfg.SOLVER.MAX_EPOCH
    print('Starting training for %g epochs...' % epochs)
    train_dataloader, test_dataloader = dataloader

    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        flg = False
        train_epoch(model, train_dataloader, solver, rtm3d_loss, cfg, tb_writer, epoch)
        if cfg.test:
            test_loss = test_epoch(model, test_dataloader, rtm3d_loss, cfg)
            if min_loss > test_loss:
                flg = True
                min_loss = test_loss
        arguments['epoch'] = epoch + 1
        checkpointer.save(model_name.format(epoch + 1), **arguments)
        if flg:
            arguments['min_loss'] = min_loss.cpu().item()
            checkpointer.save(model_best, **arguments)

    print('Finished training.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Training")
    parser.add_argument("--model-config", default="", help="specific model config path")
    parser.add_argument('--num-workers', default=20, type=int,
                        help='the num of threads for data.')
    parser.add_argument('--test', action='store_true',
                        help='run test.')
    args = parser.parse_args()
    model, dataloader, solver, rtm3d_loss, cfg = setup(args)
    train(model, dataloader, solver, rtm3d_loss, cfg)
