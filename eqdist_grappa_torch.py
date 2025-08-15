#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.fft import fft2, ifft2, fftshift, ifftshift

def ifft2c(x, axes=(-2, -1)):
    x = ifftshift(x, dim=axes)
    x = ifft2(x, dim=axes, norm="ortho")
    x = fftshift(x, dim=axes)
    return x

def fft2c(x, axes=(-2, -1)):
    x = ifftshift(x, dim=axes)
    x = fft2(x, dim=axes, norm="ortho")
    x = fftshift(x, dim=axes)
    return x

def GRAPPA_calibrate_weights_2d_torch(calibration_data_kxkyc, acc_factors_2d, 
                                block_size=(4, 4), regularization_factor=0.001, coil_axis=-1):
    block_size1, block_size2 = block_size
    acc_factor1, acc_factor2 = acc_factors_2d
    block_size1 = torch.ceil(torch.tensor(block_size1 / 2) * 2).to(torch.int64)
    block_size2 = torch.ceil(torch.tensor(block_size2 / 2) * 2).to(torch.int64)

    calibration_data_kxkyc = torch.movedim(calibration_data_kxkyc, coil_axis, -1)
    mat_size1, mat_size2, Ncoil = calibration_data_kxkyc.shape
    margin_top_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_bottom_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_left_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    margin_right_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    targetdim1_range = torch.arange(margin_top_dim1 - 1, mat_size1 - margin_bottom_dim1 + 1)
    targetdim2_range = torch.arange(margin_left_dim2 - 1, mat_size2 - margin_right_dim2 + 1)

    GRAPPA_weights = torch.zeros((acc_factor1 * acc_factor2 - 1, Ncoil, block_size1 * block_size2 * Ncoil), dtype=torch.complex64)

    for iCoil in range(Ncoil):
        for iType in range(1, acc_factor1 * acc_factor2):
            y, x = divmod(iType, acc_factor1)  # Adjusted for PyTorch compatibility
            # Create a meshgrid for indexing
            I, J = torch.meshgrid(targetdim1_range, targetdim2_range, indexing='ij')
            TargetLines = calibration_data_kxkyc[I, J, iCoil:iCoil+1].reshape(-1)

            SourceLines_thisPattern = torch.zeros((block_size1, block_size2, len(targetdim1_range), len(targetdim2_range), Ncoil), dtype=torch.complex64)
            for iBlock in range(block_size1):
                for iColumn in range(block_size2):
                    iBlock_offset = -x - acc_factor1 * (block_size1 // 2 - 1) + (iBlock) * acc_factor1
                    iColumn_offset = -y - acc_factor2 * (block_size2 // 2 - 1) + (iColumn) * acc_factor2
                    I, J = torch.meshgrid(targetdim1_range + iBlock_offset, targetdim2_range + iColumn_offset, indexing='ij')
                    
                    SourceLines_thisPattern[iBlock, iColumn] = calibration_data_kxkyc[I,J,:]

            SourceMatrix_thisPattern = SourceLines_thisPattern.permute(2, 3, 0, 1, 4).reshape(-1, block_size1 * block_size2 * Ncoil)
            
            # print(SourceMatrix_thisPattern.shape,TargetLines.shape)
            
            # L2 norm regularization
            A = SourceMatrix_thisPattern
            # print(A[:3,:3])
            # print(f"{TargetLines[:10]=}")
            AHA = A.conj().T @ A
            I = torch.eye(AHA.shape[0], dtype=torch.complex64)
            scaled_reg_factor = regularization_factor * torch.trace(AHA) / AHA.shape[0]
            coefficient = torch.linalg.solve(AHA + I * scaled_reg_factor, A.conj().T @ TargetLines)
            GRAPPA_weights[iType - 1, iCoil] = coefficient

    return GRAPPA_weights

def getGrappaImageSpaceCoilCoeff_2d_torch(block_size1, block_size2, mat_size1, mat_size2, 
                                    acc_factor1, acc_factor2, GRAPPA_weights):
    Ncoil = GRAPPA_weights.shape[1]
    new_weights_full_sumPattern = torch.zeros((mat_size1, mat_size2, Ncoil, Ncoil), dtype=torch.complex64)
    center_ky = mat_size1 // 2
    center_kx = mat_size2 // 2
    new_weights = GRAPPA_weights.view(acc_factor1 * acc_factor2 - 1, Ncoil, block_size1, block_size2, Ncoil)
    new_weights = new_weights.permute(0, 2, 3, 1, 4)

    ky2use_closest2Lastsampled = torch.arange(center_ky + 1 + acc_factor1 * (block_size1 // 2 - 1), center_ky - acc_factor1 * (block_size1 // 2), -acc_factor1)
    kx2use_closest2Lastsampled = torch.arange(center_kx + 1 + acc_factor2 * (block_size2 // 2 - 1), center_kx - acc_factor2 * (block_size2 // 2), -acc_factor2)

    for iTypes in range(acc_factor1 * acc_factor2 - 1):
        y, x = divmod(iTypes + 1, acc_factor1)
        shift_relative2firstType_dim1 = x - 1
        shift_relative2firstType_dim2 = y - 1
        ky2use = ky2use_closest2Lastsampled + shift_relative2firstType_dim1
        kx2use = kx2use_closest2Lastsampled + shift_relative2firstType_dim2
        I, J = torch.meshgrid(ky2use, kx2use, indexing='ij')
        new_weights_full_sumPattern[I, J] += torch.squeeze(new_weights[iTypes])

    for iCoil in range(Ncoil):
        new_weights_full_sumPattern[center_ky, center_kx, iCoil, iCoil] = 1

    GrappaUnmixingMap = ifft2c(new_weights_full_sumPattern,(0,1)) * torch.sqrt(torch.tensor(mat_size1 * mat_size2))
    return GrappaUnmixingMap

def GRAPPA_interpolate_imageSpace_2d_torch(undersampled_kspace_kxkyc, acc_factors_2d, block_size, 
                                     GRAPPA_weights, unmixing_map_coilWise=None, coil_axis=-1):
    
    undersampled_kspace_kxkyc = torch.movedim(undersampled_kspace_kxkyc, coil_axis, -1)
    acc_factor1, acc_factor2 = acc_factors_2d
    mat_size1, mat_size2, Ncoil = undersampled_kspace_kxkyc.shape
    if unmixing_map_coilWise is None:
        print('Recalculating GRAPPA unmixing map...')
        unmixing_map_coilWise = getGrappaImageSpaceCoilCoeff_2d_torch(block_size[0], block_size[1], 
                                                                mat_size1, mat_size2, acc_factor1, acc_factor2, 
                                                                GRAPPA_weights)

    
    # firstAcquirePoint_ky = (torch.abs(undersampled_kspace_kxkyc[:, :, 0]) > 0).nonzero()[0, 0]
    # firstAcquirePoint_kx = (torch.abs(undersampled_kspace_kxkyc[:, :, 0]) > 0).nonzero()[0, 0]
    
    firstAcquirePoint_ky = torch.nonzero(torch.sum(torch.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=1))[0][0]
    firstAcquirePoint_kx = torch.nonzero(torch.sum(torch.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=0))[0][0]
    
    # print(firstAcquirePoint_ky,firstAcquirePoint_kx)

    recon_kspace_kxkyc = torch.zeros_like(undersampled_kspace_kxkyc)
    recon_kspace_kxkyc[firstAcquirePoint_ky::acc_factor1, firstAcquirePoint_kx::acc_factor2, :] = undersampled_kspace_kxkyc[firstAcquirePoint_ky::acc_factor1, firstAcquirePoint_kx::acc_factor2, :]

    I_aliased = ifft2c(recon_kspace_kxkyc,(0,1))
    I_coils = torch.zeros_like(I_aliased)

    for ii in range(Ncoil):
        I_coils[:, :, ii] = torch.sum(I_aliased * unmixing_map_coilWise[:, :, ii, :], dim=2)

    
    recon_kspace_kxkyc = fft2c(I_coils, (0,1))
    recon_kspace_kxkyc[undersampled_kspace_kxkyc != 0] = undersampled_kspace_kxkyc[undersampled_kspace_kxkyc != 0]
    image_coilcombined_sos = torch.sqrt(torch.sum(torch.abs(ifft2c(recon_kspace_kxkyc,(0,1)))**2, dim=2))
    
    recon_kspace_kxkyc = torch.movedim(recon_kspace_kxkyc, -1, coil_axis)
    return recon_kspace_kxkyc, image_coilcombined_sos, unmixing_map_coilWise


def GRAPPA_interpolate_kSpace_2d_torch(undersampled_kspace_kxkyc, acc_factors_2d, block_size, grappa_weights, coil_axis=-1):
    """
    Interpolates missing k-space data using 2D GRAPPA for equidistant undersampling in PyTorch.
    
    Args:
        undersampled_kspace_kxkyc (Tensor): The undersampled k-space data.
        acc_factors_2d (tuple): Acceleration factors in the two dimensions.
        block_size (tuple): Block size in the two dimensions.
        grappa_weights (Tensor): Precomputed GRAPPA weights for interpolation.
    
    Returns:
        tuple: A tuple containing:
            - image_recon_sos (Tensor): The reconstructed image using Sum of Squares.
            - kspace_coils (Tensor): The interpolated k-space data.
    """
    undersampled_kspace_kxkyc = torch.movedim(undersampled_kspace_kxkyc, coil_axis, -1)
    
    acc_factor1, acc_factor2 = acc_factors_2d
    block_size1, block_size2 = block_size
    mat_size1, mat_size2, Ncoil = undersampled_kspace_kxkyc.shape

    # Define padding margins
    margin_top = acc_factor1 * (block_size1 // 2 + 1)
    margin_bottom = margin_top
    margin_left = acc_factor2 * (block_size2 // 2 + 1)
    margin_right = margin_left

    # Pad data
    padded_data = torch.nn.functional.pad(undersampled_kspace_kxkyc, 
                                          ( 0, 0, margin_left, margin_right, margin_top, margin_bottom), 
                                          mode='constant', value=0)

    # Identify the first acquired point on each axis
    first_acquired_ky = torch.nonzero(torch.sum(torch.abs(undersampled_kspace_kxkyc[:, :, 0]), dim=1))[0][0]
    first_acquired_kx = torch.nonzero(torch.sum(torch.abs(undersampled_kspace_kxkyc[:, :, 0]), dim=0))[0][0]
    acquired_lines_dim1 = torch.arange(first_acquired_ky, mat_size1, acc_factor1)
    acquired_lines_dim2 = torch.arange(first_acquired_kx, mat_size2, acc_factor2)

    for iCoil in range(Ncoil):
        for iType in range(1, acc_factor1 * acc_factor2):
            y, x = divmod(iType, acc_factor1)
            iPattern_dim1 = x
            iPattern_dim2 = y

            target_dim1 = acquired_lines_dim1 + iPattern_dim1
            target_dim2 = acquired_lines_dim2 + iPattern_dim2

            # Create a meshgrid for indexing
            I, J = torch.meshgrid(target_dim1, target_dim2, indexing='ij')
            source_lines = torch.zeros((block_size1, block_size2, len(target_dim1), len(target_dim2), Ncoil), dtype=torch.complex64)

            for iBlock in range(block_size1):
                for iColumn in range(block_size2):
                    block_offset = -iPattern_dim1 - acc_factor1 * (block_size1 // 2 - 1) + iBlock * acc_factor1
                    column_offset = -iPattern_dim2 - acc_factor2 * (block_size2 // 2 - 1) + iColumn * acc_factor2
                    BI, BJ = torch.meshgrid(target_dim1 + block_offset, target_dim2 + column_offset, indexing='ij')
                    source_lines[iBlock, iColumn, :, :, :] = padded_data[BI, BJ, :]

            source_matrix = source_lines.permute(2, 3, 0, 1, 4).reshape(len(target_dim1) * len(target_dim2), block_size1 * block_size2 * Ncoil)
            interpolated_k_space = torch.matmul(source_matrix, grappa_weights[iType - 1, iCoil, :].flatten())
            padded_data[I, J, iCoil] = interpolated_k_space.view(len(target_dim1), len(target_dim2))

    kspace_coils = padded_data[margin_top:-margin_bottom, margin_left:-margin_right, :]
    kspace_coils[undersampled_kspace_kxkyc != 0] = undersampled_kspace_kxkyc[undersampled_kspace_kxkyc != 0]

    image_recon_sos = torch.sqrt(torch.sum(torch.abs(ifft2(kspace_coils, dim=(0, 1))) ** 2, dim=2))

    kspace_coils = torch.movedim(kspace_coils, -1, coil_axis)
    return kspace_coils, image_recon_sos
