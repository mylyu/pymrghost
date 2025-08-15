#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: eqdist_grappa.py

Description: This script contains functions for GRAPPA algorithm-based MRI image reconstruction. 
             It includes functions for calibrating 2D GRAPPA weights, and interpolating k-space data 
             based on image domain operation, which is faster than common k-space implementation.
             But this only works for Cartesian 2D data with equidistant undersampling.

Author: Mengye Lyu, Shenzhen Technology University
Email: lvmengye@gmail.com or lvmengye@sztu.edu.cn
Created Date: 24, March, 2024
Last Modified: 24, March, 2024
Version: 1.0

Requirements:
    - Python 3.x
    - NumPy library for numerical operations

Reference:
    Magn Reson Med. 2009 Jun;61(6):1425-33. doi: 10.1002/mrm.21922.
    HTGRAPPA: real-time B1-weighted image domain TGRAPPA reconstruction.
    Saybasili H, Kellman P, Griswold MA, Derbyshire JA, Guttman MA.
"""

# Import necessary libraries
import numpy as np
from numpy.fft import fft, fft2 as fft2d, ifft2 as ifft2d, ifft, ifftshift, fftshift
# https://github.com/mylyu/pymrghost/blob/main/mrfft.py
from mrfft import fft2c, ifft2c, sos

'''
or use code below
def ifftc(x, axis=- 1):
    return fftshift(ifft(ifftshift(x), axis=axis, norm="ortho"))
def fft2c(x, axes=(- 2, - 1)):
    res = fftshift(fft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res
def sos(x, axis=- 1):
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res
'''

def GRAPPA_calibrate_weights_2d(calibration_data_kxkyc, acc_factors_2d, 
                                block_size=(4,4), regularization_factor=0.001, coil_axis=-1):
    """
    Calculate 2D GRAPPA weights (both kx and ky used) with L2 norm regularization.
    
    Args:
        calibration_data_kxkyc (ndarray): Calibration data with dimensions (mat_size1, mat_size2, Ncoil).
        acc_factors_2d (tuple): Acceleration factors in the kx and ky directions.
        block_size (tuple, optional): Block size (number of points as source data) for GRAPPA weights calculation. Defaults to (4,4).
        regularization_factor (float, optional): Regularization factor for L2 norm regularization. Defaults to 0.001.
    
    Returns:
        ndarray: 2D GRAPPA weights with dimensions (acc_factor1 * acc_factor2 - 1, Ncoil, block_size1 * block_size2 * Ncoil).
    """
    block_size1, block_size2 = block_size
    acc_factor1, acc_factor2 = acc_factors_2d
    print('regularization_factor', regularization_factor)
    block_size1 = (np.ceil(block_size1 / 2) * 2).astype(int)
    block_size2 = (np.ceil(block_size2 / 2) * 2).astype(int)

    calibration_data_kxkyc = np.moveaxis(calibration_data_kxkyc, coil_axis, -1)
    
    mat_size1, mat_size2, Ncoil = calibration_data_kxkyc.shape

    margin_top_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_bottom_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_left_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    margin_right_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    targetdim1_range = np.arange(margin_top_dim1-1, mat_size1 - margin_bottom_dim1 + 1)
    targetdim2_range = np.arange(margin_left_dim2-1, mat_size2 - margin_right_dim2 + 1)

    GRAPPA_weights = np.zeros((acc_factor1 * acc_factor2 - 1, Ncoil, block_size1 * block_size2 * Ncoil), dtype=np.complex64)

    Ntargetdim1 = len(targetdim1_range)
    Ntargetdim2 = len(targetdim2_range)

    # Coefficient calculation using ACS lines
    for iCoil in range(Ncoil):
        for iType in range(1, acc_factor1 * acc_factor2):  # index away from last sample in upper left direction
            y, x = np.divmod(iType, acc_factor1)
            iPattern_dim1 = x
            iPattern_dim2 = y
            TargetLines = calibration_data_kxkyc[np.ix_(targetdim1_range, targetdim2_range, [iCoil])]
            TargetLines = TargetLines.flatten()

            SourceLines_thisPattern = np.zeros((block_size1, block_size2, Ntargetdim1, Ntargetdim2, Ncoil), 
                                               dtype=np.complex64)
            for iBlock in range(block_size1):
                for iColumn in range(block_size2):
                    iBlock_offset = -iPattern_dim1 - acc_factor1 * (block_size1 // 2 - 1) + (iBlock) * acc_factor1
                    iColumn_offset = -iPattern_dim2 - acc_factor2 * (block_size2 // 2 - 1) + (iColumn) * acc_factor2
                    SourceLines_thisPattern[iBlock, iColumn] = calibration_data_kxkyc[
                        np.ix_(targetdim1_range + iBlock_offset, targetdim2_range + iColumn_offset, range(Ncoil))]
            SourceMatrix_thisPattern = np.transpose(SourceLines_thisPattern,[2,3,0,1,4]).reshape(Ntargetdim1 * Ntargetdim2, block_size1 * block_size2 * Ncoil)
            
            # L2 norm regularize
            A = SourceMatrix_thisPattern
            AHA = A.conj().T @ A
            reduced_eye = np.diag(np.abs(np.diag(AHA)) > 0)
            n_alias = np.sum(reduced_eye)
            scaled_reg_factor = regularization_factor * np.trace(AHA) / n_alias
            coefficient = np.linalg.solve(AHA + reduced_eye * scaled_reg_factor, A.conj().T @ TargetLines)
            GRAPPA_weights[iType - 1, iCoil] = coefficient

    return GRAPPA_weights


def getGrappaImageSpaceCoilCoeff_2d(block_size1, block_size2, mat_size1, mat_size2, 
                                    acc_factor1, acc_factor2, GRAPPA_weights):
    """
    Convert GRAPPA weights into image space unmixing maps.

    Args:
        block_size1 (int): Size of the block in the first dimension.
        block_size2 (int): Size of the block in the second dimension.
        mat_size1 (int): Size of the output image in the first dimension.
        mat_size2 (int): Size of the output image in the second dimension.
        acc_factor1 (int): Acceleration factor in the first dimension.
        acc_factor2 (int): Acceleration factor in the second dimension.
        GRAPPA_weights (ndarray): GRAPPA weights.

    Returns:
        ndarray: Image space unmixing maps.

    """
    Ncoil = GRAPPA_weights.shape[1]

    new_weights_full_sumPattern = np.zeros((mat_size1, mat_size2, Ncoil, Ncoil), dtype=np.complex64)
    center_ky = mat_size1 // 2
    center_kx = mat_size2 // 2

    new_weights = np.reshape(GRAPPA_weights,(acc_factor1 * acc_factor2 - 1, Ncoil, block_size1, block_size2, Ncoil))
    new_weights = np.transpose(new_weights, (0, 2, 3, 1, 4))

    ky2use_closest2Lastsampled = np.arange(center_ky + 1 + acc_factor1 * (block_size1 // 2 - 1), center_ky - acc_factor1 * (block_size1 // 2), -acc_factor1)
    kx2use_closest2Lastsampled = np.arange(center_kx + 1 + acc_factor2 * (block_size2 // 2 - 1), center_kx - acc_factor2 * (block_size2 // 2), -acc_factor2)

    for iTypes in range(acc_factor1 * acc_factor2 - 1):
        y, x = divmod(iTypes+1, acc_factor1)
        iTypes_dim1 = x
        iTypes_dim2 = y
        shift_relative2firstType_dim1 = iTypes_dim1-1
        shift_relative2firstType_dim2 = iTypes_dim2-1
        ky2use = ky2use_closest2Lastsampled + shift_relative2firstType_dim1
        kx2use = kx2use_closest2Lastsampled + shift_relative2firstType_dim2

        new_weights_full_sumPattern[np.ix_(ky2use, kx2use)] += np.squeeze(new_weights[iTypes])

    for iCoil in range(Ncoil):
        new_weights_full_sumPattern[center_ky, center_kx, iCoil, iCoil] = 1

    GrappaUnmixingMap = ifft2c(new_weights_full_sumPattern,axes=(0,1)) * np.sqrt(mat_size1 * mat_size2)

    return GrappaUnmixingMap


def GRAPPA_interpolate_imageSpace_2d(undersampled_kspace_kxkyc, acc_factors_2d, block_size, 
                                     GRAPPA_weights, unmixing_map_coilWise=None, coil_axis=-1):
    """
    Performs GRAPPA interpolation in image space for 2D data.

    Parameters:
    - undersampled_kspace_kxkyc: numpy array
        The undersampled k-space data of shape (mat_size1, mat_size2, Ncoil).
    - acc_factors_2d: tuple
        The acceleration factors in the two dimensions (acc_factor1, acc_factor2).
    - block_size: tuple
        The block size in the two dimensions (block_size1, block_size2).
    - GRAPPA_weights: numpy array
        The GRAPPA weights used for interpolation.
    - unmixing_map_coilWise: numpy array, optional
        The coil-wise unmixing map. If not provided, it will be recalculated.

    Returns:
    - recon_kspace_kxkyc: numpy array
        The reconstructed k-space data of shape (mat_size1, mat_size2, Ncoil).
    - image_coilcombined_sos: numpy array
        The coil-combined image using sum of squares of shape (mat_size1, mat_size2).
    - unmixing_map_coilWise: numpy array
        The coil-wise unmixing map used for interpolation of shape (block_size1, block_size2, Ncoil, Ncoil).

    Reference:
    Magn Reson Med. 2009 Jun;61(6):1425-33. doi: 10.1002/mrm.21922.
    HTGRAPPA: real-time B1-weighted image domain TGRAPPA reconstruction.
    Saybasili H, Kellman P, Griswold MA, Derbyshire JA, Guttman MA.
    """
    acc_factor1, acc_factor2 = acc_factors_2d
    block_size1, block_size2 = block_size
    block_size1 = (np.ceil(block_size1 / 2) * 2).astype(int)
    block_size2 = (np.ceil(block_size2 / 2) * 2).astype(int)
    # Initialize variables
    undersampled_kspace_kxkyc = np.moveaxis(undersampled_kspace_kxkyc, coil_axis, -1)
    mat_size1, mat_size2, Ncoil = undersampled_kspace_kxkyc.shape
    if unmixing_map_coilWise is None:
        print('Recalculate GRAPPA unmixing map')
        unmixing_map_coilWise = getGrappaImageSpaceCoilCoeff_2d(block_size1, block_size2, 
                                                                mat_size1, mat_size2, acc_factor1, acc_factor2, 
                                                                GRAPPA_weights)
    else:
        print('using provided unmixing map')

    # identify the location of the first sampled point on each axis.
    firstAcquirePoint_ky = np.nonzero(np.sum(np.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=1))[0][0]
    firstAcquirePoint_kx = np.nonzero(np.sum(np.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=0))[0][0]
    # Remove ACS lines if any before deconvolution, we will fill them in later.
    ksp_tmp = undersampled_kspace_kxkyc[firstAcquirePoint_ky::acc_factor1, firstAcquirePoint_kx::acc_factor2, :]
    recon_kspace_kxkyc = np.zeros_like(undersampled_kspace_kxkyc)
    recon_kspace_kxkyc[firstAcquirePoint_ky::acc_factor1, firstAcquirePoint_kx::acc_factor2, :] = ksp_tmp

    # Deconvolution
    I_aliased = ifft2c(recon_kspace_kxkyc,axes=(0,1))

    I_coils = np.zeros((mat_size1, mat_size2, Ncoil), dtype=I_aliased.dtype)   
    for ii in range(Ncoil):
        I_coils[:, :, ii] = np.sum(I_aliased * unmixing_map_coilWise[:, :, ii, :], axis=2)

    # Forward Fourier Transform to get back to k-space
    recon_kspace_kxkyc = fft2c(I_coils,axes=(0,1))

    # Refilling ACS lines
    acquired_positions = undersampled_kspace_kxkyc != 0
    recon_kspace_kxkyc[acquired_positions] = undersampled_kspace_kxkyc[acquired_positions]

    # Coil combination using sum of squares
    image_coilcombined_sos = sos(ifft2c(recon_kspace_kxkyc), -1)
    
    # handle dimentions as input
    recon_kspace_kxkyc = np.moveaxis(recon_kspace_kxkyc, -1, coil_axis)
    
    return recon_kspace_kxkyc, image_coilcombined_sos, unmixing_map_coilWise


def GRAPPA_interpolate_kSpace_2d(undersampled_kspace_kxkyc, acc_factors_2d, block_size, grappa_weights):
    """
    Interpolates missing k-space data using 2D GRAPPA for equidistant undersampling.
    
    Args:
        undersampled_kspace_kxkyc (ndarray): The undersampled k-space data.
        acc_factors_2d (tuple): Acceleration factors in the two dimensions.
        block_size (tuple): Block size in the two dimensions.
        grappa_weights (ndarray): Precomputed GRAPPA weights for interpolation.
    
    Returns:
        tuple: A tuple containing:
            - image_recon_sos (ndarray): The reconstructed image using Sum of Squares.
            - kspace_coils (ndarray): The interpolated k-space data.
    """
    acc_factor1, acc_factor2 = acc_factors_2d
    block_size1, block_size2 = block_size
    mat_size1, mat_size2, Ncoil = undersampled_kspace_kxkyc.shape

    margin_top_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_bottom_dim1 = margin_top_dim1
    margin_left_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    margin_right_dim2 = margin_left_dim2

    padded_data = np.pad(undersampled_kspace_kxkyc, ((margin_top_dim1, margin_bottom_dim1), (margin_left_dim2, margin_right_dim2), (0, 0)), mode='constant', constant_values=0)

    first_acquired_ky = np.nonzero(np.sum(np.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=1))[0][0]
    first_acquired_kx = np.nonzero(np.sum(np.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=0))[0][0]
    acquired_lines_dim1 = np.arange(first_acquired_ky, mat_size1, acc_factor1)
    acquired_lines_dim2 = np.arange(first_acquired_kx, mat_size2, acc_factor2)

    for iCoil in range(Ncoil):
        for iType in range(1, acc_factor1 * acc_factor2):
            y, x = np.divmod(iType, acc_factor1)
            iPattern_dim1 = x
            iPattern_dim2 = y

            targetdim1_range = acquired_lines_dim1 + iPattern_dim1
            targetdim2_range = acquired_lines_dim2 + iPattern_dim2

            source_lines = np.zeros((block_size1, block_size2, len(targetdim1_range), len(targetdim2_range), Ncoil), dtype=complex)

            for iBlock in range(block_size1):
                for iColumn in range(block_size2):
                    block_offset = -iPattern_dim1 - acc_factor1 * (block_size1 // 2 - 1) + iBlock * acc_factor1
                    column_offset = -iPattern_dim2 - acc_factor2 * (block_size2 // 2 - 1) + iColumn * acc_factor2
                    indices_1 = targetdim1_range + margin_top_dim1 + block_offset
                    indices_2 = targetdim2_range + margin_left_dim2 + column_offset
                    source_lines[iBlock, iColumn, :, :, :] = padded_data[np.ix_(indices_1, indices_2, range(Ncoil))]

            source_matrix = np.transpose(source_lines, (2, 3, 0, 1, 4)).reshape(len(targetdim1_range) * len(targetdim2_range), block_size1 * block_size2 * Ncoil)
            interpolated_k_space = source_matrix @ grappa_weights[iType - 1, iCoil, :].flatten()
            padded_data[np.ix_(targetdim1_range + margin_top_dim1, targetdim2_range + margin_left_dim2, [iCoil])] = interpolated_k_space.reshape((len(targetdim1_range), len(targetdim2_range), 1))

    kspace_coils = padded_data[margin_top_dim1:-margin_bottom_dim1, margin_left_dim2:-margin_right_dim2, :]
    kspace_coils[undersampled_kspace_kxkyc != 0] = undersampled_kspace_kxkyc[undersampled_kspace_kxkyc != 0]

    image_recon_sos = np.sqrt(np.sum(np.abs(ifft2c(kspace_coils, axes=(0,1))) ** 2, axis=2))

    return kspace_coils, image_recon_sos
