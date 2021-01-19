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


class SeqDatasetReader(Dataset):
    def __init__(self, root, config, augment=None, is_training=True, split='training'):
        super(SeqDatasetReader, self).__init__()
        self._root = root
        self._config = config
        self._augment = augment
        self._classes = kitti_utils.name_2_label(config.DATASET.OBJs)
        self._relate_classes = kitti_utils.name_2_label(config.DATASET.RELATE_OBJs)
        self.is_training = is_training
        self._split = split
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
        self._is_rect = config.IS_RECT
        self._norm_params = {
            'mean_rgb': np.array(config.DATASET.MEAN, np.float32).reshape((1, 1, 3)),
            'std_rgb': np.array(config.DATASET.STD, np.float32).reshape((1, 1, 3))
        }

        self._Ks = {}
        self._Ts = {}
        self._load_seqs()

        self._transform = transforms.Compose([
            transforms.Normalize(),
            transforms.ToTensor(),
            transforms.ToNCHW()
        ])
        self._start_index = 0

    def set_start_index(self, start_index):
        self._start_index = start_index

    @property
    def samples(self):
        return self._samples

    def __len__(self):
        return len(self._samples) - self._start_index - 1

    def __getitem__(self, index):
        index += self._start_index
        assert index < len(self._samples), 'out of bounds !'
        img = self.load_image(index)
        target = ParamList((img.shape[1], img.shape[0]))
        K, T = self.load_calib_param(index)
        _labels = self.load_labels(index)
        N = 1
        if len(_labels) > 0:
            cls, noise_mask, repeats = self._transform_obj_label(_labels[index][:, 0].copy())
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

        img, target = self._apply_padding(img, target)

        # Convert
        img = np.ascontiguousarray(img)
        params = {'device': self._config.DEVICE}
        target = self._build_targets(target, build_label=(len(_labels) > 0))
        params.update(self._norm_params)
        img, target = self._transform(img, targets=target, **params)
        return img, target, index

    def _load_seqs(self):
        path = os.path.join(self._root, self._split, 'image_02')
        _seqs, _samples = [], []
        _image_indices = []
        self._samples = []
        if os.path.exists(path):
            _seqs = os.listdir(path)
            for s in _seqs:
                sf = sorted(os.listdir(os.path.join(path, s)))
                _samples += [s + '/' + f.split('.')[0] for f in sf]
                _image_indices.append(len(_samples) - 1)
            self._samples = _samples
            if self._is_rect:
                shapes = np.array([self.load_image(idx).shape[-2::-1] for idx in _image_indices], dtype=np.float32)
                m = shapes.max(axis=1)
                r = self._img_size[0] / m
                ns = r.reshape(-1, 1) * shapes
                ns_max = ns.max(axis=0)
                ns_max = np.ceil(np.round(ns_max) / 32).astype(np.int) * 32
                self._img_size = ns_max

    def load_image(self, index):
        path = os.path.join(self._root, self._split, 'image_02/', '{}.png'.format(self._samples[index]))
        img = cv2.imread(path)  # BGR
        return img

    def load_labels(self, index):
        objs = []
        if os.path.exists(os.path.join(self._root, self._split, 'label_02/')):
            path = os.path.join(self._root, self._split, 'label_02/', '{}.txt'.format(self._samples[index]))
            # process label
            K, T = self.load_calib_param(index)
            invK = cv2.invert(K, flags=cv2.DECOMP_LU)[1]
            TT = np.matmul(invK, T.reshape(-1, 1)).reshape((-1))
            with open(path) as f:
                for line in f.read().splitlines():
                    splits = line.split()
                    cls = kitti_utils.name_2_label(splits[0])
                    if cls == -1:
                        continue
                    cls = np.array([float(cls)])
                    bbox = np.array([float(splits[4]), float(splits[5]), float(splits[6]), float(splits[7])])
                    dim = np.array([float(splits[8]), float(splits[9]), float(splits[10])])  # H, W, L
                    loc = np.array([float(splits[11]), float(splits[12]) - float(splits[8]) / 2,
                                    float(splits[13])]) + TT  # x, y, z
                    alpha = np.array([float(splits[3])])
                    ry = np.array([float(splits[-1])])
                    objs.append(np.concatenate([cls, bbox, dim, alpha, ry, loc], axis=0).reshape((1, -1)))
                objs = np.concatenate(objs, axis=0)

        return objs

    def load_calib_param(self, index):
        seq = self._samples[index].split('/')[0]
        if seq in self._Ks and seq in self._Ts:
            return self._Ks[seq], self._Ts[seq]
        path = os.path.join(self._root, self._split, 'calib/', '{}.txt'.format(seq))
        with open(path) as f:
            K = [line.split()[1:] for line in f.read().splitlines() if line.startswith('P2:')]
            assert len(K) > 0, 'P2 is not included in %s' % seq
            P2 = np.array(K[0], dtype=np.float32).reshape(3, 4)
            self._Ks[seq] = P2[:3, :3]
            self._Ts[seq] = P2[:3, 3]
            return P2[:3, :3], P2[:3, 3]

    def _apply_padding(self, image, target):
        sw, sh = self._img_size
        h, w, c = image.shape
        mean_rgb = np.array(cv2.mean(image))[:3]
        nimg = np.full((sh, sw, c), mean_rgb, dtype=np.uint8)
        pad_w = int(sw - w) // 2
        pad_h = int(sh - h) // 2
        nimg[pad_h:pad_h + h, pad_w:pad_w + w] = image
        if target is not None:
            if target.has_field('bbox'):
                bbox = target.get_field('bbox')
                bbox[:, 0::2] += pad_w
                bbox[:, 1::2] += pad_h
                target.update_field('bbox', bbox)
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

    def _build_targets(self, targets, build_label=True):
        W, H = self._img_size[0] // 4, self._img_size[1] // 4
        outputs = ParamList(self._img_size, is_training=self.is_training)

        down_ratio = self._config.MODEL.DOWN_SAMPLE
        Ks = targets.get_field('K')
        Ks[:, 0:6] /= down_ratio
        outputs.add_field('K', Ks)

        outputs.copy_field(targets, ['img_id', 'class', 'noise_mask',
                                     'dimension', 'location', 'Ry', 'alpha'])
        if not build_label:
            return outputs
        bboxes = targets.get_field('bbox') / down_ratio
        locations = targets.get_field('location')
        Rys = targets.get_field('Ry')
        dimensions = targets.get_field('dimension')
        N = Rys.shape[0]
        vertexs, _, mask_3ds = kitti_utils.calc_proj2d_bbox3d(dimensions,
                                                              locations,
                                                              Rys,
                                                              Ks.reshape(-1, 3, 3))
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
                    gaussian_kernel[len(xs) // 2] = 0.9999
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
        img, target, index = zip(*batch)  # transposed
        ntarget = ParamList((None, None))
        for i, t in enumerate(target):
            id = t.get_field('img_id')
            id[:, ] = i
            t.update_field('img_id', id)
            ntarget.merge(t)
        return torch.stack(img, 0), ntarget, torch.tensor(index, dtype=torch.long)


if __name__ == '__main__':
    dr = SeqDatasetReader('./datasets/data/kitti', None)

    batch_size = min(2, len(dr))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    dataloader = torch.utils.data.DataLoader(dr,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=SeqDatasetReader.collate_fn)
    for b_img, b_target in dataloader:
        print(dr)
