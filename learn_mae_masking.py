from functools import partial
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

import numpy as np

class MAEMask():

    def __init__(self, img_size=256, patch_size=16, embed_dim=1024):

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size)
        self.patch_embed.proj = nn.Conv2d(
            in_channels=1,  # change from 3 to 1
            out_channels=256,
            kernel_size=16,
            stride=16
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)

        # Parameters
        self.img_size = img_size
        self.patch_size = patch_size
        self. num_patches = (img_size // patch_size) ** 2  # 256 patches for 256x256 with 16x16 patches
        self.mask_ratio = 0.75
        self.num_mask = int(num_patches * mask_ratio)


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore



    def mask(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        return mask

    def forward_encoder(self, x, mask_ratio):
        # image show before
        image_slice = x[0, ..., 0]
        plt.imshow(image_slice, cmap='gray')
        plt.title("Slice 0 of Train data _before_ masking")
        plt.show()

        # permutate order of dimensions (batch_size, height, width, channels)
        # to (batch_size, channels, height, width)
        # Convert numpy ndarray to torch tensor
        x_tensor = torch.from_numpy(x)
        # Permute from (N, H, W, C) to (N, C, H, W)
        x_tensor = x_tensor.permute(0, 3, 1, 2).contiguous()

        # embed patches
        x = self.patch_embed(x_tensor)

        # image show masked
        x_numpy = x.detach().numpy()
        # image_slice = x_numpy[0, ..., 0]
        image_slice = x_numpy[0, :, :]
        plt.imshow(image_slice, cmap='gray')
        plt.title("Slice 0 of Train data _after_ masking")
        plt.show()

        # add pos embed w/o cls token. We need it later for the Decoder in MAE
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        # for blk in self.blocks:
        #     x = blk(x)
        # x = self.norm(x)

        return x, mask, ids_restore

    # def forward_decoder(self, x, ids_restore):
    #     # embed tokens
    #     x = self.decoder_embed(x)
    #
    #     # append mask tokens to sequence
    #     mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
    #     x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
    #     x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    #     x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
    #
    #     # add pos embed
    #     x = x + self.decoder_pos_embed
    #
    #     # apply Transformer blocks
    #     for blk in self.decoder_blocks:
    #         x = blk(x)
    #     x = self.decoder_norm(x)
    #
    #     # predictor projection
    #     x = self.decoder_pred(x)
    #
    #     # remove cls token
    #     x = x[:, 1:, :]
    #
    #     return x

    # def forward_loss(self, imgs, pred, mask):
    #     """
    #     imgs: [N, 3, H, W]
    #     pred: [N, L, p*p*3]
    #     mask: [N, L], 0 is keep, 1 is remove,
    #     """
    #     target = self.patchify(imgs)
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6) ** .5
    #
    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    #
    #     loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    #     return loss

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    #
    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    #     return x
    #
    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1] ** .5)
    #     assert h * w == x.shape[1]
    #
    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs

    # def forward(self, imgs, mask_ratio=0.75):
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
    #     loss = self.forward_loss(imgs, pred, mask)
    #     return loss, pred, mask

