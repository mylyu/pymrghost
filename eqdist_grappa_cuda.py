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

def GRAPPA_calibrate_weights_2d_torch(calibration_data_kxkyc, acc_factors_2d, device,
                                      block_size=(4, 4), regularization_factor=0.001, coil_axis=-1):
    block_size1, block_size2 = block_size
    acc_factor1, acc_factor2 = acc_factors_2d
    block_size1 = torch.ceil(torch.tensor(block_size1 / 2, device=device) * 2).to(torch.int64)  # Moved to GPU at creation
    block_size2 = torch.ceil(torch.tensor(block_size2 / 2, device=device) * 2).to(torch.int64)  # Moved to GPU at creation

    # Move data to GPU once at the start and avoid repeated movement
    calibration_data_kxkyc = torch.movedim(calibration_data_kxkyc, coil_axis, -1).to(device)
    mat_size1, mat_size2, Ncoil = calibration_data_kxkyc.shape
    margin_top_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_bottom_dim1 = acc_factor1 * (block_size1 // 2 + 1)
    margin_left_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    margin_right_dim2 = acc_factor2 * (block_size2 // 2 + 1)
    targetdim1_range = torch.arange(margin_top_dim1 - 1, mat_size1 - margin_bottom_dim1 + 1, device=device)  # GPU tensor
    targetdim2_range = torch.arange(margin_left_dim2 - 1, mat_size2 - margin_right_dim2 + 1, device=device)  # GPU tensor

    GRAPPA_weights = torch.zeros((acc_factor1 * acc_factor2 - 1, Ncoil, block_size1 * block_size2 * Ncoil), 
                                 dtype=torch.complex64, device=device)  # Allocate on GPU

    for iCoil in range(Ncoil):
        for iType in range(1, acc_factor1 * acc_factor2):
            y, x = divmod(iType, acc_factor1)
            # Precompute meshgrid only once per loop
            I, J = torch.meshgrid(targetdim1_range, targetdim2_range, indexing='ij')
            TargetLines = calibration_data_kxkyc[I, J, iCoil:iCoil+1].reshape(-1)

            # Allocate on GPU with desired shape
            SourceLines_thisPattern = torch.zeros((block_size1, block_size2, len(targetdim1_range), len(targetdim2_range), Ncoil), 
                                                  dtype=torch.complex64, device=device)
            for iBlock in range(block_size1):
                for iColumn in range(block_size2):
                    iBlock_offset = -x - acc_factor1 * (block_size1 // 2 - 1) + (iBlock) * acc_factor1
                    iColumn_offset = -y - acc_factor2 * (block_size2 // 2 - 1) + (iColumn) * acc_factor2
                    I, J = torch.meshgrid(targetdim1_range + iBlock_offset, targetdim2_range + iColumn_offset, indexing='ij')
                    SourceLines_thisPattern[iBlock, iColumn] = calibration_data_kxkyc[I, J, :]

            # Perform reshaping directly on GPU
            SourceMatrix_thisPattern = SourceLines_thisPattern.permute(2, 3, 0, 1, 4).reshape(-1, block_size1 * block_size2 * Ncoil)

            # L2 regularization with precomputed identity matrix
            A = SourceMatrix_thisPattern
            AHA = A.conj().T @ A
            I = torch.eye(AHA.shape[0], dtype=torch.complex64, device=device)  # Create on GPU
            scaled_reg_factor = regularization_factor * torch.trace(AHA) / AHA.shape[0]
            coefficient = torch.linalg.solve(AHA + I * scaled_reg_factor, A.conj().T @ TargetLines)
            GRAPPA_weights[iType - 1, iCoil] = coefficient

    return GRAPPA_weights

def getGrappaImageSpaceCoilCoeff_2d_torch(block_size1, block_size2, mat_size1, mat_size2, 
                                          acc_factor1, acc_factor2, GRAPPA_weights, device):
    Ncoil = GRAPPA_weights.shape[1]
    new_weights_full_sumPattern = torch.zeros((mat_size1, mat_size2, Ncoil, Ncoil), dtype=torch.complex64, device=device)  # Allocate on GPU
    center_ky, center_kx = mat_size1 // 2, mat_size2 // 2
    new_weights = GRAPPA_weights.view(acc_factor1 * acc_factor2 - 1, Ncoil, block_size1, block_size2, Ncoil).permute(0, 2, 3, 1, 4)  # View and permute on GPU

    ky2use_closest2Lastsampled = torch.arange(center_ky + 1 + acc_factor1 * (block_size1 // 2 - 1), 
                                              center_ky - acc_factor1 * (block_size1 // 2), -acc_factor1, device=device)  # GPU tensor
    kx2use_closest2Lastsampled = torch.arange(center_kx + 1 + acc_factor2 * (block_size2 // 2 - 1), 
                                              center_kx - acc_factor2 * (block_size2 // 2), -acc_factor2, device=device)  # GPU tensor

    for iTypes in range(acc_factor1 * acc_factor2 - 1):
        y, x = divmod(iTypes + 1, acc_factor1)
        shift_relative2firstType_dim1, shift_relative2firstType_dim2 = x - 1, y - 1
        ky2use, kx2use = ky2use_closest2Lastsampled + shift_relative2firstType_dim1, kx2use_closest2Lastsampled + shift_relative2firstType_dim2
        I, J = torch.meshgrid(ky2use, kx2use, indexing='ij')
        new_weights_full_sumPattern[I, J] += torch.squeeze(new_weights[iTypes])

    for iCoil in range(Ncoil):
        new_weights_full_sumPattern[center_ky, center_kx, iCoil, iCoil] = 1  # Directly set values without repeated reassignment

    GrappaUnmixingMap = ifft2c(new_weights_full_sumPattern, (0, 1)) * torch.sqrt(torch.tensor(mat_size1 * mat_size2, device=device))  # Inline computation
    return GrappaUnmixingMap

def GRAPPA_interpolate_imageSpace_2d_torch(undersampled_kspace_kxkyc, acc_factors_2d, block_size, 
                                           GRAPPA_weights, device, unmixing_map_coilWise=None, coil_axis=-1):
    acc_factor1, acc_factor2 = acc_factors_2d
    mat_size1, mat_size2, Ncoil = undersampled_kspace_kxkyc.shape
    if unmixing_map_coilWise is None:
        unmixing_map_coilWise = getGrappaImageSpaceCoilCoeff_2d_torch(block_size[0], block_size[1], 
                                                                      mat_size1, mat_size2, acc_factor1, acc_factor2, 
                                                                      GRAPPA_weights, device)

    undersampled_kspace_kxkyc = torch.movedim(undersampled_kspace_kxkyc, coil_axis, -1).to(device)  # Move once to GPU
    firstAcquirePoint_ky = torch.nonzero(torch.sum(torch.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=1))[0][0]
    firstAcquirePoint_kx = torch.nonzero(torch.sum(torch.abs(undersampled_kspace_kxkyc[:, :, 0]), axis=0))[0][0]

    recon_kspace_kxkyc = torch.zeros_like(undersampled_kspace_kxkyc).to(device)
    recon_kspace_kxkyc[firstAcquirePoint_ky::acc_factor1, firstAcquirePoint_kx::acc_factor2, :] = undersampled_kspace_kxkyc[firstAcquirePoint_ky::acc_factor1, firstAcquirePoint_kx::acc_factor2, :]

    I_aliased = ifft2c(recon_kspace_kxkyc, (0, 1))
    I_coils = torch.zeros_like(I_aliased).to(device)

    for ii in range(Ncoil):
        I_coils[:, :, ii] = torch.sum(I_aliased * unmixing_map_coilWise[:, :, ii, :], dim=2)

    image_coilcombined_sos = torch.sqrt(torch.sum(torch.abs(I_coils)**2, dim=2))
    recon_kspace_kxkyc = fft2c(I_coils, (0, 1))
    recon_kspace_kxkyc[undersampled_kspace_kxkyc != 0] = undersampled_kspace_kxkyc[undersampled_kspace_kxkyc != 0]

    recon_kspace_kxkyc = torch.movedim(recon_kspace_kxkyc, -1, coil_axis)
    return recon_kspace_kxkyc, image_coilcombined_sos, unmixing_map_coilWise
