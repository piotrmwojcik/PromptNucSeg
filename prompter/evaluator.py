import io
import pickle
import torch
import numpy as np
import prettytable as pt

from eval_map import eval_map
from utils import point_nms, get_tp

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


SCORE_THRESHOLD = 0.350
DISTANCE = 12

def box_cxcywh_to_xyxy(x):
    x = x * 255
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w ), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def clip_bbox(bbox_tensor):
    # Clip bounding box coordinates to the interval [0, 255]
    bbox_tensor[:, 0].clamp_(min=0, max=255)  # xmin
    bbox_tensor[:, 1].clamp_(min=0, max=255)  # ymin
    bbox_tensor[:, 2].clamp_(min=0, max=255)  # xmax
    bbox_tensor[:, 3].clamp_(min=0, max=255)  # ymax
    return bbox_tensor


def main():
    num_classes = 5

    cls_predictions = []
    cls_annotations = []

    cls_pn, cls_tn = list(torch.zeros(num_classes) for _ in range(2))
    cls_rn = torch.zeros(num_classes)

    det_pn, det_tn = list(torch.zeros(1) for _ in range(2))
    det_rn = torch.zeros(1)

    with open('/Users/piotrwojcik/Downloads/baseline4/detr_dump/results.pkl', 'rb') as file:
        result = CPU_Unpickler(file).load()
    with open('/Users/piotrwojcik/Downloads/baseline4/detr_dump/target.pkl', 'rb') as file:
        target = CPU_Unpickler(file).load()

    for k in result.keys():
        scores, boxes, labels = result[k]
        gt_boxes, gt_labels = target[k]

        gt_boxes = box_cxcywh_to_xyxy(gt_boxes)

        gt_labels = gt_labels - 1
        labels = labels - 1

        boxes = boxes[scores >= SCORE_THRESHOLD]
        labels = labels[scores >= SCORE_THRESHOLD]
        scores = scores[scores >= SCORE_THRESHOLD]
        boxes = clip_bbox(boxes)

        centroid_x = (boxes[:, 0] + boxes[:, 2]) / 2
        centroid_y = (boxes[:, 1] + boxes[:, 3]) / 2
        cnt = torch.stack((centroid_x, centroid_y), dim=1)

        gt_centroid_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_centroid_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        cnt_gt = torch.stack((gt_centroid_x, gt_centroid_y), dim=1)

        gt_points = cnt_gt.numpy()
        gt_labels = gt_labels.numpy()
        cls_annotations.append({'points': gt_points, 'labels': labels})

        for c in range(num_classes):
            ind = (labels == c)
            category_pd_points = cnt[ind]
            category_pd_scores = scores[ind]
            category_gt_points = gt_points[gt_labels == c]
            pred_num, gd_num = len(category_pd_points), len(category_gt_points)

            cls_pn[c] += pred_num
            cls_tn[c] += gd_num
            if pred_num and gd_num:
                cls_right_nums = get_tp(category_pd_points, category_pd_scores, category_gt_points, thr=DISTANCE)
                cls_rn[c] += torch.tensor(cls_right_nums)
        det_pn += len(cnt)
        det_tn += len(gt_points)

        if len(cnt) and len(gt_points):
            det_right_nums = get_tp(cnt, scores, gt_points, thr=DISTANCE)
            det_rn += torch.tensor(det_right_nums)
    eps = 1e-7
    det_r = det_rn / (det_tn + eps)
    det_p = det_rn / (det_pn + eps)
    det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)

    det_r = det_r.cpu().numpy() * 100
    det_p = det_p.cpu().numpy() * 100
    det_f1 = det_f1.cpu().numpy() * 100

    cls_r = cls_rn / (cls_tn + eps)
    cls_p = cls_rn / (cls_pn + eps)
    cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)

    cls_r = cls_r.cpu().numpy() * 100
    cls_p = cls_p.cpu().numpy() * 100
    cls_f1 = cls_f1.cpu().numpy() * 100

    table = pt.PrettyTable()
    table.add_column('CLASS', ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"])

    table.add_column('Precision', cls_p.round(2))
    table.add_column('Recall', cls_r.round(2))
    table.add_column('F1', cls_f1.round(2))

    table.add_row(['---'] * 4)

    det_p, det_r, det_f1 = det_p.round(2)[0], det_r.round(2)[0], det_f1.round(2)[0]
    cls_p, cls_r, cls_f1 = cls_p.mean().round(2), cls_r.mean().round(2), cls_f1.mean().round(2)

    table.add_row(['Det', det_p, det_r, det_f1])
    table.add_row(['Cls', cls_p, cls_r, cls_f1])
    print(table)

    #mAP = eval_map(cls_predictions, cls_annotations, 13)[0]
    #print(f'mAP: {round(mAP * 100, 2)}')


if __name__ == '__main__':
    main()
