import numpy as np
import torch
from .rotate_overlap_diff.oriented_iou_loss import  cal_iou, cal_diou, cal_giou
from .rbbox_overlap_gpu import rbbox_iou as rbbox_iou_gpu
from .rbbox_overlap_gpu import rbbox_nms as rbbox_nms_gpu


def rbbox_iou(boxes1, boxes2, device=0):  # [x, y, w, h, a]
    boxes1 = boxes1.reshape([-1, 5]).astype(np.float32)
    boxes2 = boxes2.reshape([-1, 5]).astype(np.float32)
    ious = rbbox_iou_gpu(boxes1, boxes2, device)
    return ious

def rbbox_nms(boxes, scores, iou_thresh=0.5, device=0):
    boxes = boxes.reshape([-1, 5]).astype(np.float32)
    scores = scores.reshape([-1, 1]).astype(np.float32)
    boxes = np.c_[boxes, scores]
    keeps = rbbox_nms_gpu(boxes, iou_thresh, device)
    return keeps



def rbbox_batched_nms(boxes, scores, labels, iou_thresh=0.5):
    if len(boxes) == 0:
        return np.empty([0], dtype=np.int)
    max_coordinate = boxes[:, 0:2].max() + boxes[:, 2:4].max()
    offsets = labels * (max_coordinate + 1)
    boxes = boxes.copy()
    boxes[:, :2] += offsets[:, None]
    return rbbox_nms(boxes, scores, iou_thresh)

def angle_switch(bbox, type='a2r'): 
    if type == 'a2r':
        bbox = torch.cat([bbox[..., 0: 4], bbox[...,  -1].unsqueeze(1)/180*3.14159], dim=-1)
    else:
        raise NotImplementedError
    return bbox

def iou_obb_diff(gts, preds, type='riou'):
    gt_bboxes = angle_switch(torch.from_numpy(gts).unsqueeze(0))
    pred_bboxes = angle_switch(torch.from_numpy(preds).unsqueeze(0))
    if type == 'riou':
        import ipdb; ipdb.set_trace()
        iou, *_ = cal_iou(gt_bboxes.unsqueeze(0).cuda(), pred_bboxes.unsqueeze(0).cuda())
        linear = False
        if linear:
            iou_loss = 1 - iou
        else:
            iou_loss = - iou.clamp(min=1e-6).log()

    elif type in ['giou', 'diou']:
        riou_func = cal_giou if type == 'giou' else cal_diou
        iou_loss, iou = riou_func(gt_bboxes.unsqueeze(0), pred_bboxes.unsqueeze(0))
    else:
        raise NotImplementedError
    return iou, iou_loss


