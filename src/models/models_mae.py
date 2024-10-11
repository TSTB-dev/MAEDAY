# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from ..util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def indices_to_mask(self, mask_indices, L):
        """Convert indices to binary mask.
        Args:
            masks_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
            L (int): The total number of patches.
        Returns:
            mask (tensor): The binary mask. shape of (B, L), where L is the number of patches.
        """
        B, M = mask_indices.shape
        masks = torch.zeros(B, L, device=mask_indices.device)
        masks.scatter_(dim=1, index=mask_indices, value=True)
        inverse_masks = torch.logical_not(masks)
        return masks, inverse_masks
    
    def mask_to_indices(self, masks):
        """Convert binary mask to indices.
        Args:
            masks (tensor): The binary mask. shape of (B, L), where L is the number of patches.
        Returns:
            mask_indices (tensor): The indices of masked patches. shape of (B, M), where M is the number of masked patches.
        """
        mask_indices_ = torch.nonzero(masks, as_tuple=False)  # (L, 2)
        mask_indices = []
        for i in range(masks.shape[0]):
            mask_idx = mask_indices_[mask_indices_[:, 0] == i, 1]
            mask_indices.append(mask_idx)
        mask_indices = torch.stack(mask_indices, dim=0)
        return mask_indices

    def forward_encoder(self, x, mask_indices):
        # x: (B, 3, H, W)
        # mask_indices: (B, M), where M is the number of masked patches. Each element is the index of masked patch.
        
        # embed patches
        x = self.patch_embed(x)  # (B, L, D)
        B, L, D = x.shape
        M = mask_indices.shape[1]
        
        masks, masks_inv = self.indices_to_mask(mask_indices, L)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]    

        # masking: length -> length * mask_ratio
        x_vis = x[masks_inv].view(B, L - M, D)  # (B, V, D), V = L - M

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_vis = torch.cat((cls_tokens, x_vis), dim=1)  # (B, V+1, D)

        # apply Transformer blocks
        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)  
        # x_vis: (B, V+1, D)
        # masks: (B, L), binary map which indicates the [masked] patches
        # masks_inv: (B, L), binary map which indicates the [unmasked] patches
        return x_vis, masks, masks_inv

    def forward_decoder(self, x_vis, masks_inv, mask_indices):
        # embed tokens
        x_vis = self.decoder_embed(x_vis)  
        B, _, D = x_vis.shape
        L = masks_inv.shape[1]
        V = x_vis.shape[1] - 1  
        
        # mask to indices
        vis_indices = self.mask_to_indices(masks_inv)  # (B, V)
        mask_tokens = self.mask_token.repeat(B, L-V, 1)  # (B, M, D)
        
        # unshuffle
        x = torch.zeros(B, L, D, device=x_vis.device)  # (B, L, D)
        x.scatter_(dim=1, index=vis_indices.unsqueeze(-1).repeat(1, 1, D), src=x_vis[:, 1:, :])  
        x.scatter_(dim=1, index=mask_indices.unsqueeze(-1).repeat(1, 1, D), src=mask_tokens) 
        x = torch.cat([x_vis[:, :1, :], x], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x  # (B, L, p*p*3)

    def forward_loss(self, imgs, pred, mask, apply_loss_on_vis=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        apply_loss_on_vis: if True, apply loss on visible patches, otherwise apply loss on removed patches.
        """
        target = self.patchify(imgs)  # [N, L, p*p*3]
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if apply_loss_on_vis: 
            loss = loss.sum()
        else:
            # default: apply loss on sorely removed patches
            loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, images, mask_indices, apply_loss_on_vis=False):
        """
        Args:
            images (tensor): shape of (B, 3, H, W), preprocessed images. 
            mask_indices (tensor): shape of (B, M), where M is the number of masked pathces. Each element is the index of masked patch.
            apply_loss_on_vis (bool): if True, apply loss on visible patches, otherwise apply loss on removed patches.
        Returns:
            loss (tensor): scalar, loss value.
            pred (tensor): shape of (B, L, 3, H, W), predicted patches.
        """
        # encoder
        latent, masks, masks_inv = self.forward_encoder(images, mask_indices)  # (B, V+1, D), (B, L), (B, L)
        # decoder
        pred = self.forward_decoder(latent, masks_inv, mask_indices)  # [N, L, p*p*3]
        # loss
        loss = self.forward_loss(images, pred, masks, apply_loss_on_vis)  # scalar
        pred = self.unpatchify(pred)  # [N, 3, H, W]
        return loss, pred

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks