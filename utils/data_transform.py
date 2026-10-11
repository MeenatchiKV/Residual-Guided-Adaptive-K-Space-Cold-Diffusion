import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import fastmri
from fastmri.data import subsample, transforms, mri_data
from help_func import print_var_detail



class DataTransform_Diffusion:
    def __init__(
            self,
            mask_func,
            img_size=320,
            combine_coil=True,
            flag_singlecoil=False,
    ):
        """
        data transformation class applied on diffusion models

        Args:
            mask_func: mask function that output both unfolded mask and folded mask
            img_size: int, image_size of H, W
            combine_coil: bool, check if combined coil
            flag_singlecoil: bool, check if input is singlecoil
        """
        self.mask_func = mask_func
        self.img_size = img_size
        self.combine_coil = combine_coil  # whether to combine multi-coil imgs into a single channel
        self.flag_singlecoil = flag_singlecoil
        if flag_singlecoil:
            self.combine_coil = True

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)

        # for now assume combined coil only
        if self.combine_coil:
            image_full = fastmri.rss(image_full).unsqueeze(0)  # [1,H,W,2]

        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        # ====== Fully-sampled ===
        # img space
        image_full_abs = fastmri.complex_abs(image_full)  # [Nc,H,W]

        # ====== Under-sampled ======
        # apply mask

        # assume using same mask across image coils
        mask, mask_fold = self.mask_func()  # [1,H,W] [1,H/patch_size,W/patch_size]
        mask = torch.from_numpy(mask).float()  # [1,H,W]
        mask_fold = torch.from_numpy(mask_fold).float()  # [1,H/patch_size,W/patch_size]
        mask = mask[..., None].repeat(kspace.shape[0], 1, 1, 1)  # [Nc,H,W,2]
        masked_kspace = kspace * mask

        image_masked = fastmri.ifft2c(masked_kspace)  # [Nc,H,W,2]
        image_masked_abs = fastmri.complex_abs(image_masked)  # [Nc,H,W]
        max = torch.amax(image_masked_abs, dim=(1, 2))
        scale_coeff = 1. / max  # [Nc,]

        kspace = torch.einsum('ijkl, i -> ijkl', kspace, scale_coeff)

        return kspace, mask[0, ..., 0].unsqueeze(0), mask_fold  # [B,Nc,H,W,2]


class DataTransform_UNet:
    def __init__(
            self,
            mask_func,
            img_size=320,
            combine_coil=True,
            flag_singlecoil=False,
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.combine_coil = combine_coil  # whether to combine multi-coil imgs into a single channel
        self.flag_singlecoil = flag_singlecoil
        if flag_singlecoil:
            self.combine_coil = True

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]
        Nc = kspace.shape[0]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]

        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]

        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.squeeze(-1).squeeze(0).repeat(kspace.shape[1], 1)  # [H,W]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.squeeze(-1).squeeze(0)  # [H,W]

        image_masked = fastmri.ifft2c(masked_kspace)
        image_masked = fastmri.complex_abs(image_masked)  # [Nc,H,W]

        # ====== RSS coil combination ======
        if self.combine_coil:
            image_full = fastmri.rss(image_full, dim=0)  # [H,W]
            image_masked = fastmri.rss(image_masked, dim=0)  # [H,W]

            # img [B,1,H,W], mask [B,1,H,W]
            return image_masked.unsqueeze(0), image_full.unsqueeze(0), mask.unsqueeze(0)

        else:  # if not combine coil
            # img [B,Nc,H,W], mask [B,1,H,W]
            return image_masked, image_full, mask.unsqueeze(0)


class DataTransform_WNet:
    def __init__(
        self,
        mask_func,
        img_size=320,
        flag_singlecoil=False,
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.flag_singlecoil = flag_singlecoil

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]


        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]
        image_full = fastmri.rss(image_full, dim=0)  # [H,W]


        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.repeat(kspace.shape[0], kspace.shape[1], 1, kspace.shape[3])  # [Nc,H,W,2]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.repeat(kspace.shape[0], 1, 1, kspace.shape[3])  # [Nc,H,W,2]

        # kspace [B,Nc,H,W,2], mask [B,Nc,H,W,2], image [B,H,W]
        return masked_kspace, kspace, mask, image_full


class DataTransform_VarNet:
    def __init__(
        self,
        mask_func,
        img_size=320,
        flag_singlecoil=False,
    ):
        self.mask_func = mask_func
        self.img_size = img_size
        self.flag_singlecoil = flag_singlecoil

    def __call__(self, kspace, mask, target, data_attributes, filename, slice_num):
        if self.flag_singlecoil:
            kspace = kspace[None, ...]  # [H,W,2] to [1,H,W,2]

        # k-space, transform the data into appropriate format
        kspace = transforms.to_tensor(kspace)  # [Nc,H,W,2]

        # ====== Image reshaping ======
        # img space
        image_full = fastmri.ifft2c(kspace)  # [Nc,H,W,2]
        # center cropping
        image_full = transforms.complex_center_crop(image_full, [320, 320])  # [Nc,H,W,2]
        # resize img
        if self.img_size != 320:
            image_full = torch.einsum('nhwc->nchw', image_full)
            image_full = T.Resize(size=self.img_size)(image_full)
            image_full = torch.einsum('nchw->nhwc', image_full)
        # img to k-space
        kspace = fastmri.fft2c(image_full)  # [Nc,H,W,2]


        # ====== Fully-sampled ===
        # img space
        image_full = fastmri.complex_abs(image_full)  # [Nc,H,W]
        image_full = fastmri.rss(image_full, dim=0)  # [H,W]


        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.repeat(kspace.shape[0], kspace.shape[1], 1, kspace.shape[3])  # [Nc,H,W,2]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.repeat(kspace.shape[0], 1, 1, kspace.shape[3])  # [Nc,H,W,2]

        # kspace [B,Nc,H,W,2], mask [B,Nc,H,W,2], image [B,H,W]
        return masked_kspace, kspace, mask, image_full

class DataTransform_PhysicsDiffusion:
    def __init__(
            self,
            mask_func,
            img_size=320,
            flag_singlecoil=True,
    ):
        """
        Data transformation for Physics-Guided K-Space Diffusion.

        Designed for the single-coil Physics-KCD experiment.

        Returns:
            kspace_full:
                Fully sampled single-coil k-space
                [Nc, H, W, 2]

            masked_kspace:
                Actually acquired / undersampled k-space
                [Nc, H, W, 2]

            mask:
                Acquisition mask
                [1, H, W]

            mask_fold:
                Folded mask used by K-Space Cold Diffusion
        """

        self.mask_func = mask_func
        self.img_size = img_size
        self.flag_singlecoil = flag_singlecoil

    def __call__(
            self,
            kspace,
            mask,
            target,
            data_attributes,
            filename,
            slice_num
    ):

        # ====================================================
        # 1. Convert input to single-coil format
        # ====================================================

        if self.flag_singlecoil:

            # Input from single-coil fastMRI:
            # [H, W, 2]
            #
            # Convert to:
            # [1, H, W, 2]

            kspace = kspace[None, ...]

        # ====================================================
        # 2. Convert numpy array to torch tensor
        # ====================================================

        kspace = transforms.to_tensor(kspace)

        # Expected:
        # [Nc, H, W, 2]

        # Preserve original fully sampled k-space
        kspace_full = kspace.clone()

        # ====================================================
        # 3. Convert to image domain
        # ====================================================

        image_full = fastmri.ifft2c(kspace_full)

        # [Nc, H, W, 2]

        # ====================================================
        # 4. Center crop
        # ====================================================

        image_full = transforms.complex_center_crop(
            image_full,
            [320, 320]
        )

        # [Nc, 320, 320, 2]

        # ====================================================
        # 5. Optional resize
        # ====================================================

        if self.img_size != 320:

            image_full = torch.einsum(
                'nhwc->nchw',
                image_full
            )

            image_full = T.Resize(
                size=self.img_size
            )(image_full)

            image_full = torch.einsum(
                'nchw->nhwc',
                image_full
            )

        # ====================================================
        # 6. Convert cropped image back to k-space
        # ====================================================

        kspace_full = fastmri.fft2c(image_full)

        # [Nc, H, W, 2]

        # ====================================================
        # 7. Generate acquisition mask
        # ====================================================

        mask, mask_fold = self.mask_func()

        mask = torch.from_numpy(mask).float()

        mask_fold = torch.from_numpy(mask_fold).float()

        # mask:
        # [1, H, W]

        # ====================================================
        # 8. Expand mask to complex representation
        # ====================================================

        mask_complex = mask[..., None].repeat(
            kspace_full.shape[0],
            1,
            1,
            1
        )

        # [Nc, H, W, 2]

        # ====================================================
        # 9. Generate actual measured k-space
        # ====================================================

        masked_kspace = kspace_full * mask_complex

        # ====================================================
        # 10. Normalize
        # ====================================================

        image_masked = fastmri.ifft2c(masked_kspace)

        image_masked_abs = fastmri.complex_abs(
            image_masked
        )

        scale_coeff = 1.0 / (
            torch.amax(
                image_masked_abs,
                dim=(1, 2)
            ) + 1e-8
        )

        # Same normalization coefficient is applied
        # to both full and measured k-space.

        kspace_full = torch.einsum(
            'ijkl,i->ijkl',
            kspace_full,
            scale_coeff
        )

        masked_kspace = torch.einsum(
            'ijkl,i->ijkl',
            masked_kspace,
            scale_coeff
        )

        # ====================================================
        # 11. Return
        # ====================================================

        return (
            kspace_full,
            masked_kspace,
            mask,
            mask_fold
        )


def apply_mask(data, mask_func):
    '''
    data: [Nc,H,W,2]
    mask_func: return [Nc(1),H,W]
    '''
    # mask, _ = mask_func()
    mask_return = mask_func()
    if len(mask_return) == 1:
        mask = mask_func()
    else:
        mask, _ = mask_func()
    mask = torch.from_numpy(mask)
    mask = mask[..., None]  # [Nc(1),H,W,1]
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask
