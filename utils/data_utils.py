import torch
import numpy as np
import cv2
import math


def bbox_center(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [xc, yc] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x[:, :2]) if isinstance(
        x, torch.Tensor) else np.zeros_like(x[:, :2])
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    return y


def bbox_area(x):
    y = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # w * h
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def labels_to_class_weights(labels, nc=80):
    # Get class weights (inverse frequency) from training labels
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurences per class

    # Prepend gridpoint count (for uCE trianing)
    # gpi = ((320 / 32 * np.array([1, 2, 4])) ** 2 * 3).sum()  # gridpoints per image
    # weights = np.hstack([gpi * len(labels)  - weights.sum() * 9, weights * 9]) ** 0.5  # prepend gridpoints to start

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, nc=80, class_weights=np.ones(80)):
    # Produces image weights based on class mAPs
    n = len(labels)
    class_counts = np.array(
        [np.bincount(labels[i][:, 0].astype(np.int), minlength=nc) for i in range(n)])
    image_weights = (class_weights.reshape(1, nc) * class_counts).sum(1)
    # index = random.choices(range(n), weights=image_weights, k=1)  # weight image sample
    return image_weights


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def kitti8_classes():
    x = [0, 1, 2, 3, 4, 5, 6, 7]
    return x


def dynamic_sigma(bboxes, max_bbox_area, min_bbox_area, max_sigma=19, min_sigma=3, down_ratio=4.):
    _gaussian_scale = (max_sigma - min_sigma) / (max_bbox_area - min_bbox_area) * down_ratio ** 2
    areas = bbox_area(bboxes)
    gaussian_sigma = torch.sqrt((areas - min_bbox_area) * _gaussian_scale + min_sigma)
    gaussian_radius = gaussian_sigma * 3
    return gaussian_sigma, np.ceil(gaussian_radius)


def _compute_gaussian_radius(bboxes, min_overlap=0.7):
    height, width = np.ceil(bboxes[:, 3] - bboxes[:, 1]), np.ceil(bboxes[:, 2] - bboxes[:, 0])

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return np.concatenate([r1[:, None], r2[:, None], r3[:, None]], axis=-1).min(axis=-1)


def dynamic_radius(bboxes):
    gaussian_radius = _compute_gaussian_radius(bboxes)
    gaussian_sigma = (2 * gaussian_radius + 1) / 6
    return gaussian_sigma, np.ceil(gaussian_radius)


def gaussian2D(sigma, radius):
    '''

    :param sigma: gaussian sigma
    :param radius: gaussian radius
    :return:
    '''

    offset_x = np.arange(-radius, radius + 1, 1)
    offset_y = np.arange(-radius, radius + 1, 1)
    offset_x, offset_y = np.meshgrid(offset_x, offset_y)
    offset_y, offset_x = offset_y.flatten(), offset_x.flatten()
    gaussian_kernel = -1 * (offset_x ** 2 + offset_y ** 2) / (2 * (sigma ** 2))
    gaussian_kernel = np.exp(gaussian_kernel)  # (M, )
    return gaussian_kernel, offset_x.astype(np.int32), offset_y.astype(np.int32)
