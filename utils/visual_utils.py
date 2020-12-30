import numpy as np
import cv2
from enum import Enum
import random
from utils import model_utils
from datasets.data.kitti.devkit_object import utils as kitti_utils
cv2.setNumThreads(0)


class cv_colors(Enum):
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)
    DINGXIANG = (204, 164, 227)


KITTI_COLOR_MAP = (
    cv_colors.RED.value,
    cv_colors.GREEN.value,
    cv_colors.BLUE.value,
    cv_colors.PURPLE.value,
    cv_colors.ORANGE.value,
    cv_colors.MINT.value,
    cv_colors.YELLOW.value,
    cv_colors.DINGXIANG.value
)


def getColorMap():
    colormap = [[255, 255, 255]]
    for i in range(3 * 9 - 1):
        if i % 9 == 0:
            continue
        k = i // 9
        m = i % 9
        color = [255, 255, 255]
        color[k] = (color[k] >> m)
        colormap.append(color)
    return colormap


def cv_draw_bboxes_2d(image, bboxes_2d, label_map=None, color_map=KITTI_COLOR_MAP):
    bboxes_2d_array = bboxes_2d.numpy()
    bboxes = bboxes_2d_array.get_field('bbox')
    classes = bboxes_2d_array.get_field('class').astype(np.int)
    scores = bboxes_2d_array.get_field('score') if bboxes_2d_array.has_field('score') else np.ones_like(classes)

    for cls, score, bbox in zip(classes, scores, bboxes):
        color = color_map[cls]
        label = '{}:{:.2f}'.format(label_map[cls] if label_map is not None else cls, score)
        image = plot_one_box(bbox, image, color=color, label=label, line_thickness=2)

    return image


def cv_draw_bboxes_3d(img, bboxes_3d, label_map=None, color_map=KITTI_COLOR_MAP):
    bboxes_3d_array = bboxes_3d.numpy()
    classes = bboxes_3d_array.get_field('class').astype(np.int)
    N = len(classes)
    scores = bboxes_3d_array.get_field('score') if bboxes_3d_array.has_field('score') else np.ones((N,), dtype=np.int)
    locations = bboxes_3d_array.get_field('location')
    Rys = bboxes_3d_array.get_field('Ry')
    dimensions = bboxes_3d_array.get_field('dimension')
    Ks = bboxes_3d_array.get_field('K')
    for cls, loc, Ry, dim, score, K in zip(classes, locations, Rys, dimensions, scores, Ks):
        label = label_map[cls] if label_map is not None else cls
        cv_draw_bbox_3d(img, K.reshape((3, -1)), Ry, dim, loc, label, score, color_map[cls])
    return img


def cv_draw_bboxes_3d_kitti(img, bboxes_3d, label_map=None, color_map=KITTI_COLOR_MAP):
    bboxes_3d_array = bboxes_3d.numpy()
    classes = bboxes_3d_array.get_field('class').astype(np.int)
    N = len(classes)
    scores = bboxes_3d_array.get_field('score') if bboxes_3d_array.has_field('score') else np.ones((N,), dtype=np.int)
    locations = bboxes_3d_array.get_field('location')
    Rys = bboxes_3d_array.get_field('Ry')
    dimensions = bboxes_3d_array.get_field('dimension')
    Ks = bboxes_3d_array.get_field('K')
    proj2des, bboxes_2d, _ = kitti_utils.calc_proj2d_bbox3d(dimensions, locations, Rys, Ks.reshape(N, 3, 3))

    for cls, score, proj2d, loc, bbox in zip(classes, scores, proj2des, locations, bboxes_2d):
        label = label_map[cls] if label_map is not None else cls
        cv_draw_bbox_3d_kitti(img, label, score, proj2d.T.astype(np.int), loc,  color_map[cls], thickness=2)
        plot_one_box(bbox, img, color=color_map[cls], label=label, line_thickness=2)
    return img


def cv_draw_bbox_3d(img, proj_matrix, ry, dimension, center, cls, score, color, thickness=1):
    tl = thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # to see the corners on image as red circles
    proj_2d = model_utils.calc_proj_corners(dimension, center, ry, proj_matrix).astype(np.int32)

    outline = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2]
    outline_pt = proj_2d[outline]
    # TODO put into loop
    for i in range(len(outline) - 1):
        cv2.line(img, (outline_pt[i][0], outline_pt[i][1]), (outline_pt[i + 1][0], outline_pt[i + 1][1]), color, tl)

    front_mark = np.array([[proj_2d[0][0], proj_2d[0][1]],
                           [proj_2d[1][0], proj_2d[1][1]],
                           [proj_2d[3][0], proj_2d[3][1]],
                           [proj_2d[2][0], proj_2d[2][1]]
                           ], dtype=np.int)
    front_mark = [front_mark]

    mask = np.copy(img)
    cv2.drawContours(mask, front_mark, -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    rate = 0.7
    res = rate * img.astype(np.float) + (1 - rate) * mask.astype(np.float)
    np.copyto(img, res.astype(np.uint8))

    label = '{}:({:.2f},{:.2f},{:.2f})'.format(cls, center[0], center[1], center[2])
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    box_3d = np.array(proj_2d).min(axis=0)
    c1 = (box_3d[0], box_3d[1])
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def cv_draw_bbox_3d_kitti(img, cls, score, proj_2d, center=None, color=(0, 0, 255), thickness=1):
    tl = thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    # to see the corners on image as red circles

    outline = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 5, 1, 3, 7, 6, 2]
    outline_pt = proj_2d[outline]
    # TODO put into loop
    for i in range(len(outline) - 1):
        cv2.line(img, (outline_pt[i][0], outline_pt[i][1]), (outline_pt[i + 1][0], outline_pt[i + 1][1]), color, tl)

    front_mark = np.array([[proj_2d[0][0], proj_2d[0][1]],
                           [proj_2d[1][0], proj_2d[1][1]],
                           [proj_2d[3][0], proj_2d[3][1]],
                           [proj_2d[2][0], proj_2d[2][1]]
                           ], dtype=np.int)
    front_mark = [front_mark]

    mask = np.copy(img)
    cv2.drawContours(mask, front_mark, -1, color, thickness=cv2.FILLED, lineType=cv2.LINE_8)
    rate = 0.7
    res = rate * img.astype(np.float) + (1 - rate) * mask.astype(np.float)
    np.copyto(img, res.astype(np.uint8))

    label = '{}:({:.2f},{:.2f},{:.2f})'.format(cls, center[0], center[1], center[2]) if center is not None else \
        '{}'.format(cls)
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    box_3d = np.array(proj_2d).min(axis=0)
    c1 = (box_3d[0], box_3d[1])
    c2 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def cv_draw_bbox3d_birdview(img, bboxes_3d, scaleX=0.2, scaleY=0.2, color=(255, 0, 0)):
    if scaleY is None:
        scaleY = scaleX

    bboxes_3d_array = bboxes_3d.numpy()
    classes = bboxes_3d_array.get_field('class')
    locations = bboxes_3d_array.get_field('location')
    Rys = bboxes_3d_array.get_field('Ry')
    dimensions = bboxes_3d_array.get_field('dimension')

    for cls, loc, Ry, dim in zip(classes, locations, Rys, dimensions):
        # if cls not in [0]:
        #     continue
        cv_draw_bbox_birdview(img, Ry, dim, loc, scaleX, scaleY, color)
    return img


def cv_draw_bbox_birdview(img, ry, dim, loc, scaleX=0.2, scaleY=0.2, color=(255, 0, 0)):
    h, w, _ = img.shape
    offsetX = w / 2
    offsetY = h
    R = model_utils.rotation_matrix(ry)
    corners = model_utils.create_birdview_corners(dim, location=loc, R=R)
    # transform to pixel
    rr = np.array([1./scaleX, -1./scaleY])
    tt = np.array([offsetX, offsetY])
    corners = corners[:, 0::2] * rr + tt
    index = [0, 1, 3, 2, 0]
    for i in range(len(index) - 1):
        i0 = index[i]
        i1 = index[i+1]
        cv2.line(img, (int(corners[i0][0]), int(corners[i0][1])), (int(corners[i1][0]), int(corners[i1][1])), color,
                 thickness=2, lineType=cv2.LINE_AA)


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(
        0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                    [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return img


def cv_draw_main_kf(img, m_projs, m_scores, m_cls, color=(255, 0, 0)):
    for m_projs_i, m_scores_i, m_cls_i in zip(m_projs, m_scores, m_cls):
        cv_draw_kf(img, m_projs_i, m_scores_i, m_cls_i, color=color)


def cv_draw_vertex_kf(img, v_projs, v_scores, color=(255, 0, 0)):
    for v_projs_i, v_scores_i in zip(v_projs.reshape(-1, 2), v_scores.reshape(-1)):
        cv_draw_kf(img, v_projs_i, v_scores_i, color=color)


def cv_draw_bbox3d_rtm3d(img, m_cls, m_scores, v_projs, v_scores=None, label_map=None, color_map=KITTI_COLOR_MAP):
    if v_scores is None:
        v_scores = np.zeros_like(v_projs[..., 0])
    for m_cls_i, m_scores_i, v_projs_i, v_scores_i in zip(m_cls, m_scores, v_projs, v_scores):
        label = label_map[m_cls_i] if label_map is not None else m_cls_i
        cv_draw_bbox_3d_kitti(img, label, m_scores_i, v_projs_i.astype(np.int),
                              center=None, color=color_map[m_cls_i],
                              thickness=2)


def cv_draw_kf(img, kf, scores, cls=None, label=None, color=(255, 0, 0)):
    radius = 5
    c1 = (int(kf[0]), int(kf[1]))
    cv2.circle(img, c1, radius=radius, color=color, thickness=-1)
    # label = label if label is not None else (('%s' % cls) if cls is not None else False)
    # label = ('{}:{:.2f}'.format(label, scores)) if label else '{:.2f}'.format(scores)
    # tl = radius
    # tf = max(tl - 1, 1)  # font thickness
    # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    # c2 = c1[0] - t_size[0] // 2, c1[1] - t_size[1] - 3
    # c3 = c1[0] + t_size[0] // 2, c1[1] - t_size[1] - 3
    # cv2.rectangle(img, c2, c3, color, -1, cv2.LINE_AA)  # filled
    # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
    #             [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


