# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import timm
import copy
import torch
import numpy as np
import random
import torch.nn.functional as F

from torch import nn
from models.fpn import FPN

from models.sim_vit import vit_base_patch16, SimpleFeaturePyramid


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class Backbone(nn.Module):
    def __init__(
            self,
            backbone,
            cfg
    ):
        super(Backbone, self).__init__()

        self.backbone = backbone

        self.neck = SimpleFeaturePyramid(in_feature='outcome', out_channels=256,
                                         scale_factors=(4.0, 2.0, 1.0, 0.5), top_block=None, norm="LN", square_pad=None)

        self.neck1 = SimpleFeaturePyramid(in_feature='outcome', out_channels=256,
                                         scale_factors=[4.0], top_block=None, norm="LN", square_pad=None)

    def forward(self, images):
        x = self.backbone.forward_features(images)
        _x = {'outcome': x[list(x.keys())[0]].clone()}
        __x = x[list(x.keys())[0]].clone()
        x0 = self.neck(x)
        x1 = self.neck1(_x)

        r1 = [x0[t] for t in x0.keys()]
        r2 = [x1[t] for t in x1.keys()][0]

        return r1, r2, __x


class AnchorPoints(nn.Module):
    def __init__(self, space=16):
        super(AnchorPoints, self).__init__()
        self.space = space

    def forward(self, images):
        bs, _, h, w = images.shape
        anchors = np.stack(
            np.meshgrid(
                np.arange(np.ceil(w / self.space)),
                np.arange(np.ceil(h / self.space))),
            -1) * self.space

        origin_coord = np.array([w % self.space or self.space, h % self.space or self.space]) / 2
        anchors += origin_coord

        anchors = torch.from_numpy(anchors).float().to(images.device)
        return anchors.repeat(bs, 1, 1, 1)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop=0.1):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList()

        for n, k in zip([input_dim] + h, h):
            self.layers.append(nn.Linear(n, k))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.Dropout(drop))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class DPAP2PNet(nn.Module):
    """ This is the Proposal-aware P2PNet module that performs cell recognition """

    def __init__(
            self,
            backbone,
            num_levels,
            num_classes,
            dropout=0.1,
            space: int = 16,
            hidden_dim: int = 256,
            with_mask=False
    ):
        """
            Initializes the model.
        """
        super().__init__()
        self.backbone = backbone
        self.get_aps = AnchorPoints(space)
        self.num_levels = num_levels
        self.hidden_dim = hidden_dim
        self.with_mask = with_mask
        self.strides = [2 ** (i + 2) for i in range(self.num_levels)]

        self.deform_layer = MLP(768, hidden_dim, 2, 8, drop=dropout)

        #self.reg_head = MLP(hidden_dim, hidden_dim, 2, 2, drop=dropout)
        self.cls_head = MLP(hidden_dim, hidden_dim, 2, num_classes + 1, drop=dropout)

        self.conv = nn.Conv2d(hidden_dim * num_levels, hidden_dim, kernel_size=3, padding=1)

        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SyncBatchNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=1)
        )

    def forward(self, images, train=False):
        # extract features
        (feats, feats1, x) = self.backbone(images)

        bs = images.shape[0]

        proposals = self.get_aps(images)

        o = self.backbone.backbone.forward_defrom_features(images)

        # DPP
        feat_sizes = [torch.tensor(feat.shape[:1:-1], dtype=torch.float, device=proposals.device) for feat in feats]
        #grid = (2.0 * proposals / self.strides[0] / feat_sizes[0] - 1.0)
        #roi_features = F.grid_sample(feats[1], grid, mode='bilinear', align_corners=True)
        # roi_features2 = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
        deltas2deform = self.deform_layer(o)
        deltas2deform = deltas2deform.reshape(bs, 16, 16, 2, 2, 2).permute(0, 1, 3, 2, 4, 5).reshape(bs, 32, 32, 2)
        deformed_proposals = proposals + deltas2deform

        # print(deformed_proposals[0])

        # MSD
        roi_features = []
        for i in range(self.num_levels):
            grid = (2.0 * deformed_proposals / self.strides[i] / feat_sizes[i] - 1.0)
            roi_features.append(F.grid_sample(feats[i], grid, mode='bilinear', align_corners=True))

        roi_features = torch.cat(roi_features, 1)
        roi_features = self.conv(roi_features).permute(0, 2, 3, 1)
        #deltas2refine = self.reg_head(roi_features)
        #pred_coords = deformed_proposals + deltas2refine

        pred_logits = self.cls_head(roi_features)

        output = {
            'pred_coords': pred_coords.flatten(1, 2),
            'pred_logits': pred_logits.flatten(1, 2),
            'pred_masks': F.interpolate(
                self.mask_head(feats1), size=images.shape[2:], mode='bilinear', align_corners=True)
        }

        return output


def build_model(cfg):
    encoder = vit_base_patch16(
            drop_rate=0.0,
            drop_path_rate=0.0,
            init_values=None)

    backbone = Backbone(cfg=cfg, backbone=encoder)

    model = DPAP2PNet(
        backbone,
        num_levels=cfg.prompter.neck.num_outs,
        num_classes=cfg.data.num_classes,
        dropout=cfg.prompter.dropout,
        space=cfg.prompter.space,
        hidden_dim=cfg.prompter.hidden_dim
    )

    checkpoint = torch.load('/data/pwojcik/SimMIM/TCGA_256/checkpoint-latest.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    interpolate_pos_embed(model.backbone.backbone, checkpoint_model)

    msg = model.backbone.backbone.load_state_dict(checkpoint_model, strict=False)
    print('Loading backbone for prompter')
    print(msg)

    return model
