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
from utils.ParamList import ParamList
from datasets.data.kitti.devkit_object import utils as kitti_utils
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(message)s')

root = os.path.abspath('./')
model_name = "model_{:04d}"
model_best = 'model_best'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def configure(args):
    amp.register_float_function(torch, 'sigmoid')
    amp.register_float_function(torch, 'sigmoid_')
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
    # if cfg.channels_last:
    #     memory_format = torch.channels_last
    # else:
    #     memory_format = torch.contiguous_format
    keep_batchnorm_fp32 = None if not cfg.keep_batchnorm_fp32 else cfg.keep_batchnorm_fp32
    loss_scale = None if not cfg.loss_scale else cfg.loss_scale
    cfg.update({
        'keep_batchnorm_fp32': keep_batchnorm_fp32,
        'loss_scale': loss_scale,
        'opt_level': cfg.opt_level if cfg.opt_level != '' else None
    })
    return cfg


def setup(gpu_idx, configs):
    torch.cuda.empty_cache()
    configs.gpu_idx = gpu_idx
    device = torch.device('cpu' if configs.gpu_idx == -1 else 'cuda:{}'.format(configs.gpu_idx))
    configs.update({'DEVICE': device})
    if configs.gpu_idx != -1:
        torch.cuda.set_device(configs.gpu_idx)
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
        configs.BATCH_SIZE = int(configs.BATCH_SIZE / configs.ngpus_per_node)
        configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
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
    model.cuda(configs.gpu_idx)
    if configs.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)
    solver = Solver(model, configs)
    if configs.apex:
        model = solver.apply_apex(model)
    if configs.ema:
        solver.apply_ema(model)
    checkpointer = check_point.CheckPointer(model,
                                            save_dir=save_dir,
                                            save_to_disk=True,
                                            mode='state-dict',
                                            device=configs.DEVICE)
    configs.start_epoch = 1
    configs.max_mAP = 0

    if len(configs.TRAINING.CHECKPOINT_FILE) > 0:
        ckpt = checkpointer.load(configs.TRAINING.CHECKPOINT_FILE,
                                 use_latest=(configs.TRAINING.CHECKPOINT_MODE != 'pretrained'),
                                 load_solver=False)
        if 'epoch' in ckpt and configs.TRAINING.CHECKPOINT_MODE == 'resume':
            configs.start_epoch = ckpt['epoch'] + 1
        if 'max_mAP' in ckpt and configs.TRAINING.CHECKPOINT_MODE == 'resume':
            configs.max_mAP = ckpt['max_mAP'] if ckpt['max_mAP'] < 100 else 0
        del ckpt
    # Data Parallel
    model = model_factory.make_data_parallel(model, configs)
    checkpointer.set_solver(solver)
    if configs.TRAINING.CHECKPOINT_MODE == 'resume' and configs.SOLVER.LOAD_SOLVER:
        checkpointer.load_solver_multi_gpu(configs.TRAINING.CHECKPOINT_FILE, use_latest=True)

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
                                        split='test')[0::2]
    if logger is not None:
        logger.info('number of batches in training set: {}'.format(len(train_dataloader)))

    return model, checkpointer, (train_dataloader, test_dataloader), solver, rtm3d_loss, configs, tb_writer


def main_worker(gpu_idx, configs):
    train_ops = setup(gpu_idx, configs.clone())
    train(*train_ops)


def test_epoch(model, dataloader, configs):
    dataloader, dataset = dataloader
    nb = len(dataloader)
    model.eval()
    out_path = configs.out_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        os.mkdir(os.path.join(out_path, 'data'))
    path = os.path.join(out_path, 'data')
    os.system('rm -rf {}/*'.format(path))
    half = configs.DEVICE.type != 'cpu'
    if configs.is_master_node:
        pbar = tqdm.tqdm(dataloader, total=nb)  # progress bar
    else:
        pbar = dataloader
    for imgs, targets, paths, _, indexs in pbar:
        imgs = imgs.to(configs.DEVICE)
        img_ids = targets.get_field('img_id')
        Ks = targets.get_field('K')
        Bs = imgs.shape[0]
        NKs = [None] * Bs
        invKs = [None] * Bs
        for i in range(Bs):
            if len((img_ids == i).nonzero(as_tuple=False)) > 0:
                NKs[i] = Ks[img_ids == i][0, :]
                NKs[i] = NKs[i].to(configs.DEVICE)
                invKs[i] = NKs[i].view(-1, 3, 3).inverse()
                if half:
                    invKs[i] = invKs[i].half()

        with torch.no_grad():
            t1 = time.time()
            preds = model(imgs, invKs=invKs)[0]
            t2 = time.time()

        for pred, index in zip(preds, indexs):
            results = []
            if pred is not None:
                pred = pred.cpu().numpy()
                index = index.cpu().item()
                clses = pred[:, 0].astype(np.int32)
                alphas = pred[:, 1]
                dims = pred[:, 2:5]
                locs = pred[:, 5:8]
                locs[:, 1] += (dims[:, 0]/2)
                Rys = pred[:, 8]
                scores = pred[:, 9]
                Ks = dataset.Ks[index].reshape(-1, 3, 3)
                _, bboxes, _ = kitti_utils.calc_proj2d_bbox3d(dims, locs, Rys, Ks.repeat(len(clses), axis=0))
                for cls, alpha, bbox, dim, loc, ry, score in zip(clses, alphas, bboxes, dims, locs, Rys, scores):
                    l = ('%s ' * 15 + '%s\n') % (configs.DATASET.OBJs[cls], 0, 0, alpha, *bbox, *dim, *loc, ry, score)
                    results.append(l)
            else:
                results.append(
                    ('%s ' * 15 + '%s\n') % ('DontCare', 0, 0, -10, 0, 0, 0, 0, 0, 0, 0, -1000, -1000, 0, 0, 0))
            with open(os.path.join(configs.out_path, 'data', dataset.image_files[index] + '.txt'), 'w') as f:
                f.writelines(results)
        mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        s = ('%10s' + '%10.4g' * 2) % (mem, targets.get_field('mask').shape[0], t2 - t1)
        if configs.is_master_node:
            pbar.set_description(s)

    print('start evaluate...')
    mAP_v = kitti_utils.evaluate(os.path.join(root, 'datasets/data/kitti/training/label_2'), out_path)
    mAP = torch.tensor(mAP_v, dtype=torch.float32, device=configs.DEVICE)
    if configs.distributed:
        reduced_mAP = torch_utils.reduce_tensor(mAP.data, configs.world_size)
    else:
        reduced_mAP = mAP.data
    return reduced_mAP.cpu().item()


def train_epoch(model, dataloader, solver, rtm3d_loss, configs, tb_writer, epoch):
    train_dataloader, train_sampler = dataloader
    nb = len(train_dataloader)
    epochs = configs.SOLVER.MAX_EPOCH
    model.train()
    if configs.distributed:
        train_sampler.set_epoch(epoch)
    if configs.is_master_node:
        print(('\n' + '%10s' * 10) %
              ('Epoch', 'gpu_mem', 'MKF', 'M_OFF', 'DIM', 'DEPTH', 'ORIENT', 'total', 'targets', 'lr'))
        pbar = tqdm.tqdm(enumerate(train_dataloader), total=nb)  # progress bar
    else:
        pbar = enumerate(train_dataloader)
    mloss = torch_utils.Average(torch.zeros((6,), dtype=torch.float32, device=configs.DEVICE))
    # time1 = time.time()
    for i, (imgs, targets, paths, _, _) in pbar:  # batch -------------------------------------------------------------
        imgs = imgs.to(configs.DEVICE)
        targets = targets.to(configs.DEVICE)
        pred = model(imgs)
        # time2 = time.time()
        loss, loss_items = rtm3d_loss(pred, targets)
        # time3 = time.time()
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss_items)
            return False
        solver.optim_step(loss)
        mloss.update(loss_items)

        if configs.distributed:
            reduced_losses = torch_utils.reduce_tensor(mloss.value.data, configs.world_size)
        else:
            reduced_losses = mloss.value.data

        if configs.is_master_node:
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            mask = targets.get_field('mask')
            s = ('%10s' * 2 + '%10.4g' * 8) % (
                '%g/%g' % (epoch, epochs), mem, *reduced_losses, mask.shape[0], solver.learn_rate)
            pbar.set_description(s)

            # write tensorboard
            if tb_writer is not None:
                # Tags = ['MKF', 'M_OFF', 'DIM', 'CONF', 'DEPTH', 'ORIENT', 'total']
                Tags = ['MKF', 'M_OFF', 'DIM', 'DEPTH', 'ORIENT', 'total']
                for x, tag in zip(list(mloss.value), Tags):
                    tb_writer.add_scalar('loss/' + tag, x, epoch * nb + i)
        # time1 = time.time()
    solver.scheduler_step()
    return True


def train(model, checkpointer, dataloader, solver, rtm3d_loss, configs, tb_writer):
    train_dataloader, test_dataloader = dataloader
    arguments = {'epoch': configs.start_epoch,
                 'max_mAP': configs.max_mAP}
    epochs = configs.SOLVER.MAX_EPOCH
    if configs.is_master_node:
        print('Starting training for %g epochs...' % epochs)
    for epoch in range(configs.start_epoch, epochs + 1):  # epoch ------------------------------------------------------
        flg = False
        ret = train_epoch(model, train_dataloader, solver, rtm3d_loss, configs, tb_writer, epoch)

        if configs.ema:
            solver.update_ema()
        if not ret and configs.distributed:
            dist.destroy_process_group()
        if configs.test:
            test_model = solver.ema.model if configs.ema else model
            test_mAP = test_epoch(test_model, test_dataloader, configs)
            if configs.is_master_node:
                print('Evaluation: mAP=%s in test dataset' % test_mAP)
                print('Evaluation: current max mAP=%s in test dataset' % configs.max_mAP)
                if configs.max_mAP < test_mAP:
                    flg = True
                    configs.max_mAP = test_mAP
                    print('Evaluation: current max mAP=%s in test dataset' % test_mAP)
        arguments['epoch'] = epoch
        if configs.is_master_node:
            checkpointer.save(model_name.format(epoch), **arguments)
            if flg:
                arguments['max_mAP'] = configs.max_mAP
                checkpointer.save(model_best, **arguments)

    if tb_writer is not None:
        tb_writer.close()
    if configs.distributed:
        dist.destroy_process_group()
    print('Finished training.')
    print('Evaluation: max mAP=%s in test dataset' % configs.max_mAP)


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
    parser.add_argument("--out-path", default="./tests/tmp/best", help="specific output path")
    parser.add_argument("--ema", action='store_true', help="apply ema")
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
    ####################################################################
    ##############     Apex Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--apex', action='store_true',
                        help='enabling apex.')
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')
    parser.add_argument('--opt-level', type=str, default='')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default='')
    parser.add_argument('--loss-scale', type=str, default='')
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    main(args)
