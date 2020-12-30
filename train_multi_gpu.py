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
import os
import numpy as np
import tqdm
import cv2
from torch.utils.tensorboard import SummaryWriter
import time
from fvcore.common.config import CfgNode
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from utils import torch_utils
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

model_name = "model_{:07d}"
model_best = 'model_best'
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def configure(args):
    utils.init_seeds(20)
    cfg = config.clone()
    if len(args.model_config) > 0:
        cfg.merge_from_file(args.model_config)
    opt = CfgNode(args.__dict__)
    cfg.merge_from_other_cfg(opt)
    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed
    cfg.ngpus_per_node = torch.cuda.device_count()
    return cfg


def setup(gpu_idx, configs):
    configs.gpu_idx = gpu_idx
    device = torch.device('cpu' if configs.gpu_idx == -1 else 'cuda:{}'.format(configs.gpu_idx))
    configs.update({'DEVICE': device})
    save_dir = os.path.join(configs.TRAINING.WEIGHTS, configs.MODEL.BACKBONE)
    logs_dir = os.path.join(configs.TRAINING.LOGDIR, configs.MODEL.BACKBONE)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)
        configs.subdivisions = int(64 / configs.BATCH_SIZE / configs.ngpus_per_node)
    else:
        configs.subdivisions = int(64 / configs.BATCH_SIZE)

    configs.is_master_node = (not configs.distributed) or (
            configs.distributed and (configs.rank % configs.ngpus_per_node == 0))

    if configs.is_master_node:
        logger = logging.Logger('RTM3D')
        logger.info('>>> Created a new logger')
        logger.info('>>> configs: {}'.format(configs))
        tb_writer = SummaryWriter(log_dir=os.path.join(logs_dir, 'tensorboard'))
    else:
        logger = None
        tb_writer = None
    model = model_factory.create_model(configs)
    model.to(configs.DEVICE)
    checkpointer = check_point.CheckPointer(model,
                                            save_dir=save_dir,
                                            save_to_disk=True,
                                            mode='state-dict',
                                            device=configs.DEVICE)
    configs.start_epoch = 0
    configs.min_loss = 10000

    ckpt = {}
    if len(configs.TRAINING.CHECKPOINT_FILE) > 0:
        ckpt = checkpointer.load(configs.TRAINING.CHECKPOINT_FILE,
                                 use_latest=(configs.TRAINING.CHECKPOINT_MODE != 'pretrained'),
                                 load_solver=configs.SOLVER.LOAD_SOLVER)
        if 'epoch' in ckpt and configs.TRAINING.CHECKPOINT_MODE == 'resume':
            configs.start_epoch = ckpt['epoch'] + 1
        if 'min_loss' in ckpt and configs.TRAINING.CHECKPOINT_MODE == 'resume':
            configs.min_loss = ckpt['min_loss']
    # Data Parallel
    model = model_factory.make_data_parallel(model, configs)
    solver = Solver(model, configs)
    checkpointer.set_solver(solver)
    checkpointer.load_solver(ckpt)
    del ckpt
    rtm3d_loss = RTM3DLoss(configs)

    if configs.is_master_node:
        num_parameters = model_factory.get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    if logger is not None:
        logger.info(">>> Loading dataset & getting dataloader...")
    # Create dataloader
    train_dataloader = create_dataloader(configs.DATASET.PATH, configs,
                                         TrainAugmentation(configs.INPUT_SIZE[0], mean=configs.DATASET.MEAN),
                                         is_training=True)[:2]

    test_dataloader = create_dataloader(configs.DATASET.PATH, configs,
                                        TestTransform(configs.INPUT_SIZE[0],
                                                      mean=configs.DATASET.MEAN),
                                        is_training=True,
                                        split='test')[0]
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    return model, checkpointer, (train_dataloader, test_dataloader), solver, rtm3d_loss, configs, tb_writer


def main_worker(gpu_idx, configs):
    train_ops = setup(gpu_idx, configs.clone())
    train(*train_ops)


def test_epoch(model, dataloader, rtm3d_loss, configs):
    nb = len(dataloader)
    model.eval()
    pbar = tqdm.tqdm(enumerate(dataloader), total=nb)  # progress bar
    mloss = torch.tensor(0, dtype=torch.float32, device=configs.DEVICE)
    num = 0
    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        with torch.no_grad():
            imgs = imgs.to(configs.DEVICE)
            targets = targets.to(configs.DEVICE)
            pred = model(imgs)[1]
            loss, loss_items = rtm3d_loss(pred, targets)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return
            batch_size = imgs.shape[0]
            mloss += (loss * batch_size)
            num += batch_size

    print('test loss: %s' % (mloss / num))
    return mloss / num


def train_epoch(model, dataloader, solver, rtm3d_loss, configs, tb_writer, epoch):
    train_dataloader, train_sampler = dataloader
    nb = len(train_dataloader)
    epochs = configs.SOLVER.MAX_EPOCH
    model.train()
    if configs.distributed:
        train_sampler.set_epoch(epoch)
    if configs.is_master_node:
        print(('\n' + '%10s' * 10) % (
            'Epoch', 'gpu_mem', 'MKF', 'VFM', 'M_OFF', 'V_OFF', 'total', 'targets', 'lr', 'time'))
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=nb)  # progress bar
    else:
        pbar = enumerate(train_dataloader)
    mloss = torch.zeros((5,), dtype=torch.float32, device=configs.DEVICE)
    time1 = time.time()
    for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
        imgs = imgs.to(configs.DEVICE)
        targets = targets.to(configs.DEVICE)
        pred = model(imgs)
        time2 = time.time()
        loss, loss_items = rtm3d_loss(pred, targets)
        time3 = time.time()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return

        if i:
            mloss = (mloss + loss_items) / 2
        else:
            mloss = loss_items
        solver.step(loss)
        if configs.distributed:
            reduced_loss = torch_utils.reduce_tensor(loss.data, configs.world_size)
        else:
            reduced_loss = loss.data
        if configs.is_master_node:
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            mask = targets.get_field('mask')
            s = ('%10s' * 2 + '%10.4g' * 7 + '%10s') % (
                '%g/%g' % (epoch, epochs - 1), mem, *mloss, mask.shape[0], solver.learn_rate,
                '%.1g/%.3g' % (float(time2) - float(time1), float(time3) - float(time2)))
            pbar.set_description(s)

            # write tensorboard
            if tb_writer is not None:
                Tags = ['MKF', 'VFM', 'M_OFF', 'V_OFF', 'total']
                for x, tag in zip(list(mloss), Tags):
                    tb_writer.add_scalar('loss/' + tag, x, epoch * nb + i)
        time1 = time.time()


def train(model, checkpointer, dataloader, solver, rtm3d_loss, configs, tb_writer):
    train_dataloader, test_dataloader = dataloader
    arguments = {'epoch': configs.start_epoch,
                 'min_loss': configs.min_loss}
    epochs = configs.SOLVER.MAX_EPOCH
    if configs.is_master_node:
        print('Starting training for %g epochs...' % epochs)
    for epoch in range(configs.start_epoch, epochs + 1):  # epoch ------------------------------------------------------
        flg = False
        train_epoch(model, train_dataloader, solver, rtm3d_loss, configs, tb_writer, epoch)
        if configs.test:
            test_loss = test_epoch(model, test_dataloader, rtm3d_loss, configs)
            if configs.is_master_node and configs.min_loss > test_loss:
                flg = True
                configs.min_loss = test_loss.cpu().item()
        arguments['epoch'] = epoch + 1
        if configs.is_master_node:
            checkpointer.save(model_name.format(epoch + 1), **arguments)
            if flg:
                arguments['min_loss'] = configs.min_loss
                checkpointer.save(model_best, **arguments)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        dist.destroy_process_group()
    print('Finished training.')


def main(args):
    configs = configure(args)
    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="RTM3D Training")
    parser.add_argument("--model-config", default="", help="specific model config path")
    parser.add_argument('--test', action='store_true',
                        help='run test.')
    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='GPU index to use.')
    parser.add_argument('--num-workers', default=20, type=int,
                        help='the num of threads for data.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    main(args)
