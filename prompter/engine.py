import sys
import math
import itertools
from pathlib import Path

import prettytable as pt

#from prompter.evaluator import CPU_Unpickler, box_cxcywh_to_xyxy, clip_bbox, SCORE_THRESHOLD
from utils import *
from tqdm import tqdm
from eval_map import eval_map
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import OrderedDict


def train_one_epoch(
        args,
        model,
        train_loader,
        criterion,
        optimizer,
        epoch,
        device,
        model_ema=None,
        scaler=None
):
    model.train()
    criterion.train()

    log_info = dict()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    for data_iter_step, (images, type_map, masks, points_list, labels_list) in enumerate(
            metric_logger.log_every(train_loader, args.print_freq, header)):
        images = images.to(device)
        masks = masks.to(device)
        type_map = type_map.to(device)

        targets = {
            'gt_type_map': type_map,
            'gt_masks': masks,
            'gt_nums': [len(points) for points in points_list],
            'gt_points': [points.view(-1, 2).to(device).float() for points in points_list],
            'gt_labels': [labels.to(device).long() for labels in labels_list],
        }


        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss_dict = criterion(outputs, targets, epoch)
            losses = sum(loss for loss in loss_dict.values())

        # if epoch >= 20:
        #     for idx in range(10):
        #         image = images[idx]
        #         pd_points = outputs['pred_coords'].clone()[idx]
        #         pd_points = pd_points.detach().cpu().numpy()
        #         gt_type_mask = targets['gt_type_map'][idx]
        #
        #         assert not torch.all(gt_type_mask.int() == 0)
        #         scores = outputs['pred_logits'][idx].softmax(-1).detach().cpu().numpy()
        #         import numpy as np
        #         classes = np.argmax(scores, axis=-1)
        #         valid_flag = classes < (scores.shape[-1] - 1)
        #
        #         points = pd_points[valid_flag]
        #         rest = pd_points[~valid_flag]
        #         scores = scores[valid_flag].max(1)
        #
        #         #gt_type_mask = targets['gt_type_map'][idx]
        #
        #         image = image.permute(1, 2, 0).cpu().numpy()
        #         plt.imshow(image)
        #         #points = pd_points
        #         plt.scatter(points[:, 0], points[:, 1], c='r', marker='.', s=10)
        #         plt.scatter(rest[:, 0], rest[:, 1], c='b', marker='+', s=10)
        #         plt.savefig(f'/data/pwojcik/prompter_dump/img_{idx}.png')
        #         plt.close()
        #
        #         mask = gt_type_mask.cpu().numpy()
        #
        #         colors = ['black', 'red', 'green', 'blue', 'yellow', 'purple']
        #         cmap = mcolors.ListedColormap(colors, N=6)
        #         plt.imshow(mask, cmap=cmap)
        #         plt.axis('off')
        #         plt.savefig(f'/data/pwojcik/prompter_dump/mask_{idx}.png', bbox_inches='tight', pad_inches=0)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        for k, v in loss_dict_reduced.items():
            log_info[k] = log_info.get(k, 0) + v.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if args.clip_grad > 0:  # clip gradient
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        if model_ema and data_iter_step % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        log_info["lr"] = optimizer.param_groups[0]["lr"]

    return log_info


def visualise_prompts(img_path, gt_pt, gt_cl, pt, cl, path, limit):
    colormap = np.array(['red', 'blue', 'green', 'brown', 'yellow', 'black'])
    brighter_colormap = np.array([
        '#FF6666',  # Brighter red
        '#6666FF',  # Brighter blue
        '#66FF66',  # Brighter green
        '#CD853F',  # Brighter brown
        '#FFFF66',  # Brighter yellow
        '#FF00FF',  # Brighter magenta
        '#00FFFF'  # Brighter cyan
    ])
    mkdir(f'{path}')
    if limit < 50:
        im = plt.imread('../segmentor/' + img_path)
        implot = plt.imshow(im, cmap='gray')

        gt_prompt = gt_pt
        if len(gt_prompt) > 0:
            categories = gt_cl
            plt.scatter(gt_prompt[:, 0], gt_prompt[:, 1], s=40, c=brighter_colormap[categories], marker='o')

        a = pt
        if len(a) > 0:
            categories = cl
            plt.scatter(a[:, 0], a[:, 1], s=40, c=colormap[categories.astype(int)], marker='+')

        plt.savefig(f'{path}/{os.path.basename(img_path)}')
        plt.clf()

def load_from_pickle(path):
    with open(os.path.join(Path(path), 'results.pkl'), 'rb') as file:
        result = CPU_Unpickler(file).load()
    with open(os.path.join(Path(path), 'target.pkl'), 'rb') as file:
        target = CPU_Unpickler(file).load()

    scores, boxes, labels = result[k]
    gt_boxes, gt_labels = target[k]

    gt_boxes = box_cxcywh_to_xyxy(gt_boxes)

    gt_labels = gt_labels - 1
    labels = labels - 1

    # topk_values, topk_indexes = torch.topk(scores, 100)
    # boxes = boxes[topk_indexes]
    # labels = labels[topk_indexes]
    # scores = scores[topk_indexes]

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

    return cnt, labels, scores, cnt_gt, gt_labels


# def evaluate_from_ds(
#         cfg,
#         test_loader,
#         device,
#         epoch=0,
#         calc_map=False,
#         visualise=False,
#         vis_path='',
# ):
#         class_names = test_loader.dataset.classes
#         num_classes = len(class_names)
#
#         cls_predictions = []
#         cls_annotations = []
#
#         cls_pn, cls_tn = list(torch.zeros(num_classes).to(device) for _ in range(2))
#         cls_rn = torch.zeros(num_classes).to(device)
#
#         det_pn, det_tn = list(torch.zeros(1).to(device) for _ in range(2))
#         det_rn = torch.zeros(1).to(device)
#
#         iou_scores = []
#
#         epoch_iterator = tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
#                               dynamic_ncols=True, disable=not is_main_process())
#
#         i = 0
#         batch_size = 16
#         cnt, labels, scores, cnt_gt, gt_labels = load_from_pickle('/Users/piotrwojcik/Downloads/baseline4/detr_dump4/')
#         cnt = torch.split(cnt, batch_size)
#         labels = torch.split(labels, batch_size)
#         scores = torch.split(scores, batch_size)
#         cnt_gt = torch.split(cnt_gt, batch_size)
#         gt_labels = torch.split(gt_labels, batch_size)
#         for data_iter_step  in range(len(cnt)):
#             if data_iter_step % get_world_size() != get_rank():  # To avoid duplicate evaluation for some test samples
#                 continue
#
#             epoch_iterator.set_description(
#                 "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))
#
#             gt_points = cnt_gt[i]
#             pd_masks = None
#
#             if pd_masks is not None:
#                 masks = masks[0].numpy()
#                 intersection = (pd_masks * masks).sum()
#                 union = (pd_masks.sum() + masks.sum() + 1e-7) - intersection
#                 iou_scores.append(intersection / (union + 1e-7))
#
#             gt_points = gt_points[0].reshape(-1, 2).numpy()
#             labels = labels[0].numpy()
#
#             cls_annotations.append({'points': gt_points, 'labels': labels})
#
#             cls_pred_sample = []
#             for c in range(cfg.data.num_classes):
#                 ind = (pd_classes == c)
#                 category_pd_points = pd_points[ind]
#                 category_pd_scores = pd_scores[ind]
#                 category_gt_points = gt_points[labels == c]
#
#                 cls_pred_sample.append(np.concatenate([category_pd_points, category_pd_scores[:, None]], axis=-1))
#
#                 pred_num, gd_num = len(category_pd_points), len(category_gt_points)
#                 cls_pn[c] += pred_num
#                 cls_tn[c] += gd_num
#
#                 if pred_num and gd_num:
#                     cls_right_nums = get_tp(category_pd_points, category_pd_scores, category_gt_points, thr=cfg.test.match_dis)
#                     cls_rn[c] += torch.tensor(cls_right_nums, device=cls_rn.device)
#
#             cls_predictions.append(cls_pred_sample)
#
#             det_pn += len(pd_points)
#             det_tn += len(gt_points)
#
#             if len(pd_points) and len(gt_points):
#                 det_right_nums = get_tp(pd_points, pd_scores, gt_points, thr=cfg.test.match_dis)
#                 det_rn += torch.tensor(det_right_nums, device=det_rn.device)
#
#         if get_world_size() > 1:
#             dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
#             dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
#             dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)
#
#             dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
#             dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
#             dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)
#
#             cls_predictions = list(itertools.chain.from_iterable(all_gather(cls_predictions)))
#             cls_annotations = list(itertools.chain.from_iterable(all_gather(cls_annotations)))
#
#             iou_scores = np.concatenate(all_gather(iou_scores))
#
#         eps = 1e-7
#         det_r = det_rn / (det_tn + eps)
#         det_p = det_rn / (det_pn + eps)
#         det_f1 = (2 * det_r * det_p) / (det_p + det_r + eps)
#
#         det_r = det_r.cpu().numpy() * 100
#         det_p = det_p.cpu().numpy() * 100
#         det_f1 = det_f1.cpu().numpy() * 100
#
#         cls_r = cls_rn / (cls_tn + eps)
#         cls_p = cls_rn / (cls_pn + eps)
#         cls_f1 = (2 * cls_r * cls_p) / (cls_r + cls_p + eps)
#
#         cls_r = cls_r.cpu().numpy() * 100
#         cls_p = cls_p.cpu().numpy() * 100
#         cls_f1 = cls_f1.cpu().numpy() * 100
#
#         table = pt.PrettyTable()
#         table.add_column('CLASS', class_names)
#
#         table.add_column('Precision', cls_p.round(2))
#         table.add_column('Recall', cls_r.round(2))
#         table.add_column('F1', cls_f1.round(2))
#
#         table.add_row(['---'] * 4)
#
#         det_p, det_r, det_f1 = det_p.round(2)[0], det_r.round(2)[0], det_f1.round(2)[0]
#         cls_p, cls_r, cls_f1 = cls_p.mean().round(2), cls_r.mean().round(2), cls_f1.mean().round(2)
#
#         table.add_row(['Det', det_p, det_r, det_f1])
#         table.add_row(['Cls', cls_p, cls_r, cls_f1])
#         print(table)
#         if calc_map:
#             mAP = eval_map(cls_predictions, cls_annotations, cfg.test.match_dis)[0]
#             print(f'mAP: {round(mAP * 100, 2)}')
#
#         metrics = {'Det': [det_p, det_r, det_f1], 'Cls': [cls_p, cls_r, cls_f1],
#                    'IoU': (np.mean(iou_scores) * 100).round(2)}
#         return metrics, table.get_string()

@torch.inference_mode()
def evaluate(
        cfg,
        model,
        test_loader,
        device,
        epoch=0,
        calc_map=False,
        visualise=False,
        vis_path='',
):
    model.eval()
    class_names = test_loader.dataset.classes
    num_classes = len(class_names)

    cls_predictions = []
    cls_annotations = []

    cls_pn, cls_tn = list(torch.zeros(num_classes).to(device) for _ in range(2))
    cls_rn = torch.zeros(num_classes).to(device)

    det_pn, det_tn = list(torch.zeros(1).to(device) for _ in range(2))
    det_rn = torch.zeros(1).to(device)

    iou_scores = []

    epoch_iterator = tqdm(test_loader, file=sys.stdout, desc="Test (X / X Steps)",
                          dynamic_ncols=True, disable=not is_main_process())

    i = 0
    for data_iter_step, (images, gt_points, labels, inst_mask, masks, ori_shape, img_path) in enumerate(epoch_iterator):
        assert len(images) == 1, 'batch size must be 1'

        if data_iter_step % get_world_size() != get_rank():  # To avoid duplicate evaluation for some test samples
            continue

        epoch_iterator.set_description(
            "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))

        images = images.to(device)

        pd_points, pd_scores, pd_classes, pd_masks = predict(
            model,
            images,
            data_iter_step=data_iter_step,
            ori_shape=ori_shape[0].numpy(),
            filtering=cfg.test.filtering,
            nms_thr=cfg.test.nms_thr,
            visualise=visualise
        )
        print('!!!')
        print(pd_points.shape)
        print(pd_scores.shape)
        print(labels.shape)

        if visualise:
            visualise_prompts(img_path[0], gt_points[0], labels[0], pd_points, pd_classes, vis_path, i)
            i += 1

        if pd_masks is not None:
            masks = masks[0].numpy()
            intersection = (pd_masks * masks).sum()
            union = (pd_masks.sum() + masks.sum() + 1e-7) - intersection
            iou_scores.append(intersection / (union + 1e-7))

        gt_points = gt_points[0].reshape(-1, 2).numpy()
        labels = labels[0].numpy()

        cls_annotations.append({'points': gt_points, 'labels': labels})

        cls_pred_sample = []
        for c in range(cfg.data.num_classes):
            ind = (pd_classes == c)
            category_pd_points = pd_points[ind]
            category_pd_scores = pd_scores[ind]
            category_gt_points = gt_points[labels == c]

            cls_pred_sample.append(np.concatenate([category_pd_points, category_pd_scores[:, None]], axis=-1))

            pred_num, gd_num = len(category_pd_points), len(category_gt_points)
            cls_pn[c] += pred_num
            cls_tn[c] += gd_num

            if pred_num and gd_num:
                cls_right_nums = get_tp(category_pd_points, category_pd_scores, category_gt_points, thr=cfg.test.match_dis)
                cls_rn[c] += torch.tensor(cls_right_nums, device=cls_rn.device)

        cls_predictions.append(cls_pred_sample)

        det_pn += len(pd_points)
        det_tn += len(gt_points)

        if len(pd_points) and len(gt_points):
            det_right_nums = get_tp(pd_points, pd_scores, gt_points, thr=cfg.test.match_dis)
            det_rn += torch.tensor(det_right_nums, device=det_rn.device)

    if get_world_size() > 1:
        dist.all_reduce(det_rn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(det_pn, op=dist.ReduceOp.SUM)

        dist.all_reduce(cls_pn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_tn, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_rn, op=dist.ReduceOp.SUM)

        cls_predictions = list(itertools.chain.from_iterable(all_gather(cls_predictions)))
        cls_annotations = list(itertools.chain.from_iterable(all_gather(cls_annotations)))

        iou_scores = np.concatenate(all_gather(iou_scores))

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
    table.add_column('CLASS', class_names)

    table.add_column('Precision', cls_p.round(2))
    table.add_column('Recall', cls_r.round(2))
    table.add_column('F1', cls_f1.round(2))

    table.add_row(['---'] * 4)

    det_p, det_r, det_f1 = det_p.round(2)[0], det_r.round(2)[0], det_f1.round(2)[0]
    cls_p, cls_r, cls_f1 = cls_p.mean().round(2), cls_r.mean().round(2), cls_f1.mean().round(2)

    table.add_row(['Det', det_p, det_r, det_f1])
    table.add_row(['Cls', cls_p, cls_r, cls_f1])
    print(table)
    if calc_map:
        mAP = eval_map(cls_predictions, cls_annotations, cfg.test.match_dis)[0]
        print(f'mAP: {round(mAP * 100, 2)}')

    metrics = {'Det': [det_p, det_r, det_f1], 'Cls': [cls_p, cls_r, cls_f1],
               'IoU': (np.mean(iou_scores) * 100).round(2)}
    return metrics, table.get_string()
