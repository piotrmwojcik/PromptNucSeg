import sys
import math
import itertools
import torch
from pathlib import Path
from torchvision.utils import save_image
import prettytable as pt

from utils import *
from tqdm import tqdm
from eval_map import eval_map
from typing import Sequence, Tuple, Dict, Optional, Union
from collections import OrderedDict


Point = Tuple[int, int]
PointsPerImage = Sequence[Point]

done_overlays = False

def save_random_overlays(
    images: torch.Tensor,            # [B, 3, H, W], values in [0,1] (or normalized; see denorm)
    masks: torch.Tensor,             # [B, H, W] or [B, 1, H, W], binary/float
    points: Optional[Union[Sequence[PointsPerImage], Dict[int, PointsPerImage]]] = None,
    out_dir: str = "debug_overlays",
    k: int = 8,
    alpha: float = 0.5,
    color=(0.0, 0.0, 1.0),           # mask overlay color (R,G,B) in [0,1]
    dot_radius: int = 3,
    point_color=(1.0, 0.0, 0.0),     # dot color (R,G,B) in [0,1]
    point_format: str = "yx",        # "yx" (row,col) or "xy"
    mean=None, std=None              # optional: denorm (e.g., ImageNet)
):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    B, C, H, W = images.shape
    assert C == 3, "Expected 3-channel images"

    # De-normalize (optional)
    imgs = images.detach().float().cpu()
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=imgs.dtype).view(1,3,1,1)
        std_t  = torch.tensor(std,  dtype=imgs.dtype).view(1,3,1,1)
        imgs = imgs * std_t + mean_t

    # Prep region masks
    m = masks
    if m.ndim == 3:
        m = m.unsqueeze(1)  # [B,1,H,W]
    m = (m.detach().float().cpu() > 0.5).float()

    # Colors / alpha
    color_t = torch.tensor(color, dtype=imgs.dtype).view(1,3,1,1)
    dot_col = torch.tensor(point_color, dtype=imgs.dtype).view(3,1,1)
    a = float(alpha)

    # Helper to draw a filled circle at (y,x)
    def draw_dot(overlay: torch.Tensor, y: int, x: int, radius: int):
        y0 = max(0, y - radius); y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius); x1 = min(W, x + radius + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy = torch.arange(y0, y1)
        xx = torch.arange(x0, x1)
        Y, X = torch.meshgrid(yy, xx, indexing="ij")
        mask = ((Y - y)**2 + (X - x)**2) <= radius**2
        patch = overlay[0, :, y0:y1, x0:x1]
        overlay[0, :, y0:y1, x0:x1] = torch.where(mask.unsqueeze(0), dot_col, patch)

    # Iterator over points for image i
    def iter_points_for(i: int):
        if points is None:
            return []
        if isinstance(points, dict):
            pts = points.get(i, [])
        else:
            # expect sequence of length B -> per-image lists
            pts = points[i] if len(points) == B else []
        # convert to (y,x)
        for p in pts:
            if p is None or len(p) != 2:
                continue
            y, x = (p[1], p[0]) if point_format.lower() == "xy" else (p[0], p[1])
            if 0 <= y < H and 0 <= x < W:
                yield int(y), int(x)

    # Pick random indices
    k = min(k, B)
    idx = torch.randperm(B)[:k]

    for i in idx.tolist():
        im = imgs[i:i+1]           # [1,3,H,W]
        ms = m[i:i+1]              # [1,1,H,W]
        overlay = (im * (1 - a*ms)) + (color_t * a * ms)
        overlay = overlay.clamp(0, 1)

        # Draw points
        for y, x in iter_points_for(i):
            draw_dot(overlay, x, y, dot_radius)

        # Save outputs
        save_image(im,       out / f"img_{i:03d}.png")
        save_image(ms,       out / f"mask_{i:03d}.png")
        save_image(overlay,  out / f"overlay_{i:03d}.png")

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

    for data_iter_step, (images, masks, points_list, labels_list) in enumerate(
            metric_logger.log_every(train_loader, args.print_freq, header)):
        images = images.to(device)
        masks = masks.to(device)

        targets = {
            'gt_masks': masks,
            'gt_nums': [len(points) for points in points_list],
            'gt_points': [points.view(-1, 2).to(device).float() for points in points_list],
            'gt_labels': [labels.to(device).long() for labels in labels_list],
        }

        #print('!!! ', images.shape, targets['gt_masks'].shape)
        #print('!!! ', targets['gt_points'][0].shape)
        #if epoch == 0 and data_iter_step == 2:
        #    save_random_overlays(images, targets['gt_masks'], targets['gt_points'], out_dir="overlays",
        #                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(images)
            loss_dict = criterion(outputs, targets, epoch)
            losses = sum(loss for loss in loss_dict.values())

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

    return log_info


@torch.inference_mode()
def evaluate(
        cfg,
        model,
        test_loader,
        device,
        epoch=0,
        calc_map=False,
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

    for data_iter_step, (images, gt_points, labels, masks, ori_shape) in enumerate(epoch_iterator):
        assert len(images) == 1, 'batch size must be 1'

        if data_iter_step % get_world_size() != get_rank():  # To avoid duplicate evaluation for some test samples
            continue

        epoch_iterator.set_description(
            "Epoch=%d: Test (%d / %d Steps) " % (epoch, data_iter_step, len(test_loader)))

        images = images.to(device)

        pd_points, pd_scores, pd_classes, pd_masks = predict(
            model,
            images,
            ori_shape=ori_shape[0].numpy(),
            filtering=cfg.test.filtering,
            nms_thr=cfg.test.nms_thr,
        )

        if pd_masks is not None:
            masks = masks[0].numpy()
            intersection = (pd_masks * masks).sum()
            union = (pd_masks.sum() + masks.sum() + 1e-7) - intersection
            iou_scores.append(intersection / (union + 1e-7))

        gt_points = gt_points[0].reshape(-1, 2).numpy()
        #gt_points = gt_points[0].reshape(-1, 2)[:, [1, 0]].numpy()
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
