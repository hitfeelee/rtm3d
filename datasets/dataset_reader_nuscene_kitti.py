
from torch.utils.data import Dataset
import numpy as np
import random
import os
import cv2
import torch
from utils.ParamList import ParamList
from preprocess import transforms
from torch.utils.data import DataLoader
from datasets.data.kitti.devkit_object import utils as kitti_utils
from utils import data_utils
import sys
from utils import torch_utils
import math


class DatasetReader(Dataset):
    def __init__(self, root, config, augment=None, is_training=True, split='train'):
        super(DatasetReader, self).__init__()
        self._split = split
        self._root = root
        self._config = config
        self._augment = augment
        self._classes = kitti_utils.name_2_label(config.DATASET.OBJs)
        self._relate_classes = kitti_utils.name_2_label(config.DATASET.RELATE_OBJs)
        self.is_training = is_training
        self._aug_params = {
            'hsv_h': config.DATASET.aug_hsv_h,
            'hsv_s': config.DATASET.aug_hsv_s,
            'hsv_v': config.DATASET.aug_hsv_v,
            'degrees': config.DATASET.aug_degrees,
            'translate': config.DATASET.aug_translate,
            'scale': config.DATASET.aug_scale,
            'shear': config.DATASET.aug_shear,
        }
        self._img_size = [config.INPUT_SIZE[0]] * 2
        self._is_mosaic = config.IS_MOSAIC
        self._is_rect = config.IS_RECT
        self._norm_params = {
            'mean_rgb': np.array(config.DATASET.MEAN, np.float32).reshape((1, 1, 3)),
            'std_rgb': np.array(config.DATASET.STD, np.float32).reshape((1, 1, 3))
        }

        with open(os.path.join(root, 'ImageSets', '{}_pure.txt'.format(self._split))) as f:
            self._image_files = f.read().splitlines()
            # self._image_files = sorted(self._image_files)

        label_file = os.path.join(root, 'cache', 'label_{}.npy'.format(self._split))  # saved labels in *.npy file
        self._labels = np.load(label_file, allow_pickle=True)
        k_file = os.path.join(root, 'cache', 'k_{}.npy'.format(self._split))  # saved labels in *.npy file
        self._K = np.load(k_file, allow_pickle=True)
        assert len(self._image_files) == len(self._labels) == len(self._K), 'Do not match labels and images'

        sp = os.path.join(root, 'cache', 'shape_{}.npy'.format(self._split))  # shapefile path
        s = np.load(sp, allow_pickle=True)
        self.__shapes = s
        self._image_files, self._labels, s, self._K = self._clean_bad_data_and_shuffle(work_num=40)
        self._samples = len(self._image_files) // config.BATCH_SIZE * config.BATCH_SIZE
        s = np.array(s).astype(dtype=np.int)
        self.__shapes = s
        if self._is_rect:
            m = s.max(axis=1)
            r = self._img_size[0] / m
            ns = r.reshape(-1, 1) * s
            ns_max = ns.max(axis=0)
            ns_max = np.ceil(np.round(ns_max) / 32).astype(np.int) * 32
            self._img_size = ns_max

        self._transform = transforms.Compose([
            transforms.Normalize(),
            # transforms.ToPercentCoords(),
            # transforms.ToXYWH(),
            transforms.ToTensor(),
            transforms.ToNCHW()
        ])
        # self._remove_val()

    def _clean_bad_data_and_shuffle(self, work_num=20):
        classes = self._classes

        def multi_thread_process(files, labels, shapes, Ks):
            n_files = []
            n_labels = []
            n_shapes = []
            n_Ks = []
            for i, label in enumerate(labels):
                if len(label) == 0:
                    continue
                if any([cls in classes for cls in label[:, 0]]):
                    n_files.append(files[i])
                    n_labels.append(label)
                    n_shapes.append(shapes[i])
                    n_Ks.append(Ks[i])
            return n_files, n_labels, n_shapes, n_Ks

        N = math.ceil(len(self._image_files) / work_num)
        th_pools = []
        for i in range(work_num):
            start = i * N
            end = min((i + 1) * N, len(self._image_files))
            args = [self._image_files[start:end], self._labels[start:end], self.__shapes[start:end], self._K[start:end]]
            th = torch_utils.MultiThread(i, multi_thread_process, args)
            th_pools.append(th)

        for i in range(work_num):
            th_pools[i].start()
        for i in range(work_num):
            th_pools[i].join()

        files = []
        labels = []
        shapes = []
        Ks = []
        for i in range(work_num):
            f, l, s, k = th_pools[i].get_result()
            files += f
            labels += l
            shapes += s
            Ks += k
        indices = [i for i in range(len(files))]
        np.random.seed(200)
        np.random.shuffle(indices)
        return [files[i] for i in indices], [labels[i] for i in indices], \
               [shapes[i] for i in indices], [Ks[i] for i in indices]

    def _remove_val(self):
        for f in self._image_files:
            import os
            path = os.path.join('./tests/tmp/best/data', '{}.txt'.format(f))
            os.system('rm {}'.format(path))
        print('remove success !')

    @property
    def labels(self):
        return self._labels

    @property
    def shapes(self):
        return self.__shapes

    @property
    def Ks(self):
        return self._K

    @property
    def image_files(self):
        return self._image_files

    def __len__(self):
        return self._samples #len(self._labels)

    def __getitem__(self, index):
        indices = [index]
        if self._is_mosaic and self.is_training:
            indices += [random.randint(0, len(self._labels) - 1) for _ in range(3)]  # 3 additional image indices
        images = []
        targets = []

        for i, idx in enumerate(indices):
            img = self.load_image(idx)
            target = ParamList((img.shape[1], img.shape[0]))
            K = self._K[idx]
            _labels = self._labels[idx].copy()
            cls, noise_mask, repeats = self._transform_obj_label(self._labels[idx][:, 0].copy())
            _labels = np.repeat(_labels, repeats=repeats, axis=0)
            N = len(cls)
            target.add_field('class', cls)
            target.add_field('img_id', np.zeros((N,), dtype=np.int))
            target.add_field('bbox', _labels[:, 1:5])
            target.add_field('dimension', _labels[:, 5:8])
            target.add_field('alpha', _labels[:, 8])
            target.add_field('Ry', _labels[:, 9])
            target.add_field('location', _labels[:, -3:])
            mask = np.ones((N,), dtype=np.int)
            mask[cls == -1] = 0
            target.add_field('mask', mask)
            target.add_field('noise_mask', noise_mask)
            target.add_field('K', np.repeat(K.copy().reshape(1, 9), repeats=N, axis=0))
            if self._augment is not None:
                img, target = self._augment(img, targets=target, **self._aug_params)
            images.append(img)
            targets.append(target)
        if self._is_mosaic and self.is_training:
            img, target = self._apply_mosaic(images, targets)
        else:
            img, target = self._apply_padding(images, targets)

        # Convert
        img = np.ascontiguousarray(img)
        params = {'device': self._config.DEVICE}
        target = self._build_targets(target)
        params.update(self._norm_params)
        img, target = self._transform(img, targets=target, **params)
        path = os.path.join(self._root, self._split, 'image_2/{}.png'.format(self._image_files[index]))
        return img, target, path, self.__shapes[index], index

    def load_image(self, index):
        path = os.path.join(self._root, self._split, 'image_2/', '{}.png'.format(self._image_files[index]))
        img = cv2.imread(path)  # BGR
        return img

    def _load_calib_param(self, index):
        path = os.path.join(self._root, self._split, 'calib/', '{}.txt'.format(self._image_files[index]))
        with open(path) as f:
            K = [line.split()[1:] for line in f.read().splitlines() if line.startswith('P2:')]
            assert len(K) > 0, 'P2 is not included in %s' % self._image_files[index]
            return np.array(K[0], dtype=np.float32)

    def _apply_mosaic(self, images, targets):
        assert len(images) == 4 and len(targets) == 4
        sw, sh = self._img_size
        c = images[0].shape[2]
        sum_rgb = np.zeros([images[0].ndim, ])
        for img in images:
            sum_rgb += np.array(cv2.mean(img))[:3]
        mean_rgb = sum_rgb / len(images)
        img4 = np.full((sh * 2, sw * 2, c), mean_rgb, dtype=np.uint8)  # base image with 4 tiles
        offsets = [(0, 0), (sw, 0), (0, sh), (sw, sh)]
        target4 = ParamList((sw, sh))
        for i, img, target in zip(range(4), images, targets):
            h, w, _ = img.shape
            pad_w = int(sw - w) // 2
            pad_h = int(sh - h) // 2
            y_st = pad_h + offsets[i][1]
            x_st = pad_w + offsets[i][0]
            img4[y_st:y_st + h, x_st:x_st + w] = img
            bbox = target.get_field('bbox')
            bbox[:, 0::2] += x_st
            bbox[:, 1::2] += y_st
            target.update_field('bbox', bbox)
            np.clip(bbox[:, 0::2], 0, 2 * sw, out=bbox[:, 0::2])  # use with random_affine
            np.clip(bbox[:, 1::2], 0, 2 * sh, out=bbox[:, 1::2])
            target4.merge(target)

        raff = transforms.RandomAffine2D()

        # img4 = cv2.resize(img4, (sw, sh), interpolation=cv2.INTER_LINEAR)
        param = {
            'border': (-sh//2, -sw//2)
        }
        param.update(self._aug_params)
        return raff(img4, target4, **param)

    def _apply_padding(self, images, targets):
        img = images[0]
        sw, sh = self._img_size
        target = targets[0]
        h, w, c = img.shape
        mean_rgb = np.array(cv2.mean(img))[:3]
        nimg = np.full((sh, sw, c), mean_rgb, dtype=np.uint8)
        pad_w = int(sw - w) // 2
        pad_h = int(sh - h) // 2
        bbox = target.get_field('bbox')
        bbox[:, 0::2] += pad_w
        bbox[:, 1::2] += pad_h
        target.update_field('bbox', bbox)
        nimg[pad_h:pad_h + h, pad_w:pad_w + w] = img
        if target.has_field('K'):
            K = target.get_field('K')
            K[:, 2] += pad_w
            K[:, 5] += pad_h
            target.update_field("K", K)

        return nimg, target

    def _transform_obj_label(self, src_label):
        repeats = []
        dst_labels = []
        noise_mask = []
        for label in src_label:
            dst_label = self._classes.index(int(label)) if label in self._classes else -1
            if dst_label == -1:
                dst_label = [k for k, re_labels in enumerate(self._relate_classes) if label in re_labels]
                dst_label, mask = (dst_label, [1] * len(dst_label)) if len(dst_label) > 0 else ([-1], [0])
                dst_labels += dst_label
                repeats += [len(dst_label)]
                noise_mask += mask
            else:
                dst_labels += [dst_label]
                repeats += [1]
                noise_mask += [0]
        return np.array(dst_labels), np.array(noise_mask), repeats

    def _build_targets(self, targets):
        W, H = self._img_size[0] // 4, self._img_size[1] // 4
        outputs = ParamList(self._img_size, is_training=self.is_training)
        outputs.copy_field(targets, ['img_id', 'class', 'noise_mask',
                                     'dimension', 'location', 'Ry', 'alpha'])
        down_ratio = self._config.MODEL.DOWN_SAMPLE
        bboxes = targets.get_field('bbox') / down_ratio
        locations = targets.get_field('location')
        Rys = targets.get_field('Ry')
        dimensions = targets.get_field('dimension')
        Ks = targets.get_field('K')
        N = Rys.shape[0]
        Ks[:, 0:6] /= down_ratio
        vertexs, _, mask_3ds = kitti_utils.calc_proj2d_bbox3d(dimensions,
                                                              locations,
                                                              Rys,
                                                              Ks.reshape(-1, 3, 3),
                                                              xrange=(-200, self._img_size[0]+200),
                                                              yrange=(-200, self._img_size[1]+200))
        vertexs = np.ascontiguousarray(np.transpose(vertexs, axes=[0, 2, 1]))
        centers = vertexs[:, -1]
        vertexs = vertexs[:, :-1]
        m_projs = centers.astype(np.long)
        m_offs = centers - m_projs
        outputs.add_field('m_proj', m_projs)
        outputs.add_field('m_off', m_offs)
        m_masks = (m_projs[:, 0] >= 0) & (m_projs[:, 0] < W) & (m_projs[:, 1] >= 0) & (m_projs[:, 1] < H)
        m_masks &= mask_3ds
        outputs.add_field('mask', m_masks)
        v_projs = vertexs.astype(np.long)
        v_coor_offs = vertexs - centers.reshape(-1, 1, 2)
        v_masks = (v_projs[..., 0] >= 0) & (v_projs[..., 0] < W) & (v_projs[..., 1] >= 0) & (v_projs[..., 1] < H)
        outputs.add_field('vertex', vertexs)
        outputs.add_field('v_coor_off', v_coor_offs)
        outputs.add_field('v_mask', v_masks)
        outputs.add_field('K', Ks)
        # for bbox3d vertexs loss training
        # invK = np.linalg.inv(Ks.reshape(-1, 3, 3))
        # outputs.add_field('INVK', invK)

        if self._config.DATASET.GAUSSIAN_GEN_TYPE == 'dynamic_radius':
            gaussian_sigma, gaussian_radius = data_utils.dynamic_radius(bboxes)
        else:
            gaussian_sigma, gaussian_radius = data_utils.dynamic_sigma(bboxes,
                                                                       self._config.DATASET.BBOX_AREA_MAX,
                                                                       self._config.DATASET.BBOX_AREA_MIN)
        clses = targets.get_field('class')
        num_cls = len(self._classes)
        noise_masks = targets.get_field('noise_mask')
        m_hm = np.zeros((num_cls, H, W), dtype=np.float)
        for i in range(N):
            m_mask = m_masks[i]
            noise_mask = noise_masks[i]
            if m_mask:
                # to-do
                gaussian_kernel, xs, ys = data_utils.gaussian2D(gaussian_sigma[i], gaussian_radius[i])
                if noise_mask:
                    gaussian_kernel[len(xs) // 2] = 0.9995 if self._config.apex else 0.9999
                m_proj = m_projs[i]
                cls = clses[i]
                m_xs = xs + m_proj[0]
                m_ys = ys + m_proj[1]
                valid = (m_xs >= 0) & (m_xs < W) & (m_ys >= 0) & (m_ys < H)
                m_hm[cls, m_ys[valid], m_xs[valid]] = np.maximum(m_hm[cls, m_ys[valid], m_xs[valid]],
                                                                 gaussian_kernel[valid])
        outputs.add_field('m_hm', np.expand_dims(m_hm, axis=0))
        return outputs

    @staticmethod
    def collate_fn(batch):
        img, target, path, shape, index = zip(*batch)  # transposed
        ntarget = ParamList((None, None))
        for i, t in enumerate(target):
            id = t.get_field('img_id')
            id[:, ] = i
            t.update_field('img_id', id)
            ntarget.merge(t)
        # ntarget.to_tensor()
        return torch.stack(img, 0), ntarget, path, shape, torch.tensor(index, dtype=torch.long)


def create_dataloader(path, cfg, transform=None, is_training=False, split='train'):
    dr = DatasetReader(path, cfg, augment=transform, is_training=is_training, split=split)
    batch_size = min(cfg.BATCH_SIZE, len(dr))
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 20])  # number of workers
    nw = cfg.num_workers
    sampler = None
    if hasattr(cfg, 'distributed') and cfg.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dr)

    data_loader = torch.utils.data.DataLoader(dr,
                                              batch_size=batch_size,
                                              num_workers=nw,
                                              pin_memory=True,
                                              shuffle=(sampler is None),
                                              collate_fn=DatasetReader.collate_fn,
                                              sampler=sampler)
    return data_loader, sampler, dr


if __name__ == '__main__':
    dr = DatasetReader('./datasets/data/kitti', None)

    batch_size = min(2, len(dr))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dr,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=DatasetReader.collate_fn)
    for b_img, b_target in dataloader:
        print(dr)
