import json

import cv2
from pycocotools.coco import COCO
from pycocotools.mask import iou, frPyObjects
import numpy as np
import torch
from cvtools import cv_load_image


def vis_prop(fn, boxes, scores, labels):
    def add_object_on_image(image, box, score, posi=True):
        b = list(map(int, box))
        text = '{:.4f}'.format(score)
        c = (255, 0, 0) if posi else (0, 0, 255)
        cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), c, thickness=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        tag_font_scale = 0.3
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font, tag_font_scale, 1)
        txt_tl = b[0], b[1] + int(1. * txt_h)

        cv2.putText(image, text, txt_tl, font, tag_font_scale, c, lineType=cv2.LINE_AA)
        return image

    image = cv_load_image(fn)

    for b, s, l in zip(boxes, scores, labels):
        image = add_object_on_image(image, b, s, l > 0)

    cv2.imshow('prop', image)
    cv2.waitKey()


def scale_bbox(index, fname, bbox, label, scale=0., is_xywh=False):
    bbox = bbox.copy()
    if is_xywh:
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    bbox[0] = int(max(0, bbox[0] - w * scale))
    bbox[1] = int(max(0, bbox[1] - h * scale))
    bbox[2] = int(bbox[2] + w * scale)
    bbox[3] = int(bbox[3] + w * scale)
    bbox = list(map(int, bbox))
    ploy = '{},{},{},{},{},{},{},{}'.format(bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3])
    data = '{}\t{}\t{}\t{}\n'.format(index, fname, ploy, label)
    return data


def make_pth_data():
    gt_path = '/core/data/dataset/biaozhu/yf_new_zkzh/mg_val_cls.json'
    gt = COCO(gt_path)
    pth_path = '/data4t/data/zhangyufei/expi/zkzh_cls/inference/mg_val_cls/predictions_step1.pth'
    dt = torch.load(pth_path)

    fg_thresh = 0.5
    bg_thresh = 0.5

    recog_data = []
    cnt = 0
    for image_id, bbox in dt.items():
        if len(bbox) == 0:
            continue
        image = gt.loadImgs(image_id)[0]
        fname = image['file_name']
        anno_ids = gt.getAnnIds(image_id)
        anno = gt.loadAnns(anno_ids)
        bbox_gt = np.array([i['bbox'] for i in anno])
        bboxes_dt = bbox.bbox.numpy().copy()
        bboxes_dt_xywh = bboxes_dt.copy()
        bboxes_dt_xywh[:, 2] -= bboxes_dt_xywh[:, 0]
        bboxes_dt_xywh[:, 3] -= bboxes_dt_xywh[:, 1]
        # iou xywh
        overlaps = iou(bboxes_dt_xywh, bbox_gt, [False] * len(bbox_gt))
        if isinstance(overlaps, list):
            overlaps = np.zeros(len(bboxes_dt_xywh))
        print(overlaps)
        bboxes = []
        labels = []
        scores = []
        for overlap_val, bbox in zip(overlaps, bboxes_dt):
            if bbox[0] <= 5 or bbox[1] <= 5 or bbox[2] >= image['width'] - 5 or bbox[3] >= image['height'] - 5:
                continue
            overlap = overlap_val.max()
            if bg_thresh < overlap < fg_thresh:
                continue
            if overlap >= fg_thresh:
                label = 1
                cnt += 1
            elif overlap <= bg_thresh:
                label = 0

            bbox = bbox.copy()
            bbox = list(map(int, bbox))
            bboxes.append(bbox)
            labels.append(label)
            scores.append(overlap)
            ploy = '{},{},{},{},{},{},{},{}'.format(
                bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3])
            data = '{}\t{}\t{}\t{}\n'.format(image_id, fname, ploy, label)
            recog_data.append(data)
        print(labels)
        print(len(labels))
        print(image_id)
        vis_prop(fname, bboxes, scores, labels)

    print('pos: ', cnt)
    print('neg: ', len(recog_data) - cnt)
    with open('/data4t/data/zhangyufei/jet/code/fix_box.log', 'w') as f:
        f.writelines(recog_data)

    # img0 = cv_load_image(fname)
    # orig_img0 = img0.copy()
    # print(bbox)
    # b = bbox.bbox.numpy().tolist()
    # print('num of proposal: ', len(b))
    # for c in b:
    #     cv2.rectangle(img0, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), (0, 0, 255), 1)
    # cv2.imshow('test', img0)
    # cv2.waitKey(0)


def make_dt_data():
    recog_data = []

    gt_path = '/core/data/dataset/biaozhu/yf_new_zkzh/mg_zkzh_8t.json'
    gt = COCO(gt_path)
    dt_path = '/data4t/data/zhangyufei/expi/zkzh11-1/inference/mg_zkzh_8t/bbox.json'
    with open(dt_path) as f:
        dt = json.load(f)

    cnt = 0
    # step1: 加入刀的gt
    pos = neg = 0
    image_ids = gt.getImgIds()
    for image_id in image_ids:
        image = gt.loadImgs(image_id)[0]
        fname = image['file_name']
        anno_ids = gt.getAnnIds(image_id)
        annotations = gt.loadAnns(anno_ids)
        for anno in annotations:
            if anno['category_id'] == 1:
                label = 1
                pos += 1
            else:
                label = 0
                neg += 1
            cnt += 1
            data = scale_bbox(cnt, fname, anno['bbox'], label, scale=0., is_xywh=True)
            recog_data.append(data)
    print('1st:  pos:{} neg:{}'.format(pos, neg))

    # step2: 加入刀的dt
    pos = neg = 0
    # record pos_dt
    for ind, prediction in enumerate(dt):
        image_id = prediction['image_id']
        bbox_dt = [prediction['bbox']]
        image = gt.loadImgs(image_id)[0]
        fname = image['file_name']
        anno_ids = gt.getAnnIds(image_id)
        anno = gt.loadAnns(anno_ids)
        bbox_gt = [i['bbox'] for i in anno]
        overlap = iou(bbox_dt, bbox_gt, [False] * len(bbox_gt))
        # gt is none -> overlap -> []
        if isinstance(overlap, list):
            overlap = 0
        else:
            overlap = overlap.max()
        if overlap >= 0.5:
            label = 1
            pos += 1
            cnt += 1
            data = scale_bbox(cnt, fname, bbox_dt[0], label, scale=0., is_xywh=True)
            recog_data.append(data)
    print('2.5th:  pos:{} neg:{}'.format(pos, neg))
    # record neg_dt
    for ind, prediction in enumerate(dt):
        image_id = prediction['image_id']
        bbox_dt = [prediction['bbox']]
        image = gt.loadImgs(image_id)[0]
        fname = image['file_name']
        anno_ids = gt.getAnnIds(image_id)
        anno = gt.loadAnns(anno_ids)
        bbox_gt = [i['bbox'] for i in anno]
        overlap = iou(bbox_dt, bbox_gt, [False] * len(bbox_gt))
        if isinstance(overlap, list):
            overlap = 0
        else:
            overlap = overlap.max()
        if overlap <= 0:
            label = 0
            neg += 1
            cnt += 1
            data = scale_bbox(cnt, fname, bbox_dt[0], label, scale=0., is_xywh=True)
            recog_data.append(data)
    print('2nd:  pos:{} neg:{}'.format(pos, neg))

    with open('/core7/data/zhangyufei/data/txt_data/8t_cls.txt', 'w') as f:
        f.writelines(recog_data)


if __name__ == '__main__':
    make_dt_data()
