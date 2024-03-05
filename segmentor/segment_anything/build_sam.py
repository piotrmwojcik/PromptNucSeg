# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from functools import partial
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .modeling.sim_vit import vit_base_patch16


def build_sam_vit_h(cfg):
    return _build_sam(
        cfg,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(cfg):
    return _build_sam(
        cfg,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
    )


def build_sam_vit_b(cfg):
    return _build_sam(
        cfg,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


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


def _build_sam(
        cfg,
        encoder_embed_dim,
        encoder_depth,
        encoder_num_heads,
        encoder_global_attn_indexes,
):
    prompt_embed_dim = 256
    image_size = cfg.segmentor.img_size
    vit_patch_size = cfg.segmentor.patch_size

    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=vit_base_patch16(
            drop_rate=0.0,
            drop_path_rate=0.0,
            init_values=None),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_classes=cfg.data.num_classes,
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        num_classes=cfg.data.num_classes,
        multimask=cfg.segmentor.multimask
    )
    sam.eval()

    if cfg.segmentor.type.endswith("B"):
        ckpt = 'pretrained/sam_vit_b_01ec64.pth'
    elif cfg.segmentor.type.endswith("L"):
        ckpt = 'pretrained/sam_vit_l_0b3195.pth'
    elif cfg.segmentor.type.endswith("H"):
        ckpt = 'pretrained/sam_vit_h_4b8939.pth'
    else:
        raise NotImplementedError(f"Unknown model type: {cfg.segmentor.type}")

    with open(ckpt, "rb") as f:
        pretrained_state_dict = torch.load(f, map_location='cpu')

        model_state_dict = sam.state_dict()
        updated_state_dict = {k: v for k, v in pretrained_state_dict.items() if
                              k in model_state_dict and not k.startswith("image_encoder.")}
        model_state_dict.update(updated_state_dict)

        sam.load_state_dict(model_state_dict)

        checkpoint = torch.load('/data/pwojcik/SimMIM/TCGA_256/checkpoint-latest.pth', map_location='cpu')
        interpolate_pos_embed(self.encoder, checkpoint_model)
        checkpoint_model = checkpoint['model']

        msg = sam.image_encoder.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    return sam
