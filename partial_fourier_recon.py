#!/usr/bin/env python
# coding: utf-8
"""
MRI Partial Fourier Reconstruction using POCS
---------------------------------------------

Description:
    This Python package implements an iterative POCS (Projections Onto Convex Sets) 
    method to reconstruct partial Fourier (PF) MRI data.
Usage example:
    kspFull_kxkycz, kspZpad_kxkycz = pf_recon_pocs_ms2d(ksp,10);
Workflow:
    `pf_recon_pocs_ms2d` will call `pocs_pf`, which does as follows.
    1. Determine Input Data Type:
        - find the echo peak and zero-fill the data matrix.
    
    2. Low Resolution Image Creation:
        - Generate a low-resolution image for each channel/coil.
        - Utilize a symmetrically sampled section around the central k-space.
        - Use a Hamming filter to prevent Gibbs-Ringing.

    3. Save Phase of Low-Resolution Images:
        - Preserve the phase for POCS reconstruction, assuming phase variations
          are from coil sensitivities and B0-inhomogeneities.

    4. Apply Reference Phase in Image Space:
        - Transform zero-filled data to image space using inverse FFT.
        - Remove the existing phase and apply the reference phase map.
        - Convert back to k-space using FFT.
        - Iteratively refine until k-space points are filled appropriately.

Note:
    Avoiding aliasing is crucial for the effectiveness of POCS.

Authorship:
    Original MATLAB code by Martin Blaimer, Uvo Hoelscher, and Michael Völker.
    Translated and adapted to Python by Mengye Lyu.

Contact:
    Dr. Mengye Lyu
    Shenzhen Technology University
    Email: lvmengye@sztu.edu.cn

For issues, suggestions or improvements, kindly reach out to the email mentioned above.
"""
import numpy as np
try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn, fftshift, ifftshift
except ImportError:
    from scipy.fft import fftn, ifftn, fftshift, ifftshift
        
from scipy.signal.windows import hann

def pf_recon_pocs_ms2d(ksp_kxkycz, iter, full_nX=None):
    """
    pf_recon_pocs_ms2d function only accepts 2d multislice data in [kx, ky, coil, slice]
    kx should be the dim which pf is applied.
    """
    # Get dimensions
    nX, nY, nC, nZ = ksp_kxkycz.shape

    # Handle the case where full_Nx is not provided
    if full_nX is None:
        data_sum = np.sum(np.abs(ksp_kxkycz), axis=(1,2,3))
        I = np.argmax(data_sum)
        full_nX = 2*((nX - I))
        print(f"auto detect peak at {I}; full Nx is {full_nX}")
        if full_nX <= nX:
            raise ValueError('data is not partial fourier')

    # Zero-pad
    kspZpad_kxkycz = np.zeros((full_nX, nY, nC, nZ), dtype=complex)
    kspZpad_kxkycz[-nX:] = ksp_kxkycz

    # Permute dimensions
    kspZpad_ckxkyz = np.transpose(kspZpad_kxkycz, (2,0,1,3))

    kspFull_ckxkyz = np.zeros((nC, full_nX, nY, nZ), dtype=complex)

    for iSlice in range(nZ):
        if (iSlice+1) % 5 == 0:  # +1 because Python uses 0-based indexing
            print(f"processing slice {iSlice+1}/{nZ}")

        # Call to pocs_tidy, assuming the function has been translated
        kspFull_ckxkyz[:,:,:,iSlice] = pocs_pf(kspZpad_ckxkyz[:,:,:,iSlice], iter).reshape((nC,full_nX,nY))

        # If you are using the pocs_nd function, uncomment the following line and ensure it's translated
        # kspFull_ckxkyz[:,:,:,iSlice] = pocs_nd(kspZpad_ckxkyz[:,-nX:,:,iSlice], iter, 1)

    print("done")

    # Reverse the permutation
    kspFull_kxkycz = np.transpose(kspFull_ckxkyz, (1,2,0,3))

    return kspFull_kxkycz, kspZpad_kxkycz

def pocs_pf(kspaceInput, iter):
    """
    Partial-Fourier Reconstruction with POCS
    Allowed shapes for kspaceInput are...
%                 ... Ny x Nx (not tested)
%                 ... Nc x Ny x Nx (tested!)
%                 ... Nc x Ny x Nx x Nz (not tested)
%
%               With Nc == number of receive Channels / Coils.
    """
    Ndim = kspaceInput.ndim
    # print(f"{kspaceInput.flags=}")
    # Check for the right dimensionality of the input data
    if Ndim > 4 or Ndim < 2:
        raise ValueError("First input 'kspace' should have one of these shapes: Ny x Nx, Nc x Ny x Nx, or Nc x Ny x Nx x Nz")

    # Handle 2D data case by reshaping it to have a dummy channel dimension
    if Ndim == 2:
        kspaceInput = np.expand_dims(kspaceInput, axis=0)  # 1 x Ny x Nx
        wasAddedCoilDim = True
        Ndim = 3
    else:
        wasAddedCoilDim = False

    sz = kspaceInput.shape[1:]

    # Detect sampling pattern and check if there are zeros (undersampling)
    samplingPattern = np.any(kspaceInput != 0, axis=0).astype(int)
    if np.count_nonzero(samplingPattern) == np.prod(sz):
        raise ValueError("You must insert zeros to the matrix first")

    # We'll replace detectPFdim with the Python version once we have it
    pfDim, isUpper, isLower, numSamples = detectPFdim(samplingPattern, False)

    if len(sz) < 3:
        sz = (sz[0], sz[1], 1)

    # % Reverse the entries if the first ones are zero-filled 
    subs = [slice(None)] * Ndim  # Equivalent to {':', ':', ':', ':'}

    if isLower:
        subs[pfDim+1] = slice(-1, -sz[pfDim]-1, -1)  # Reverse along the specified dimension
        kspaceInput = kspaceInput[tuple(subs)]  # This is k-space!
        subs[pfDim+1] = slice(0, sz[pfDim])  # lalala, we didn't do anything...
    #print(subs)

    # We'll replace findSymSampled with the Python version once we have it
    centreLine, idxSym = findSymSampled(kspaceInput, pfDim, numSamples)
    szSym = len(idxSym)
    prec = kspaceInput.dtype

    # Replace ndWindowFilter with the Python version
    filter = ndWindowFilter(sz, pfDim, szSym, idxSym)
    filter = np.expand_dims(filter, axis=0)
    if filter.ndim==4 and filter.shape[-1]==1:
        kspaceInput = kspaceInput[...,np.newaxis]
    # Apply the low-pass filter
    kspLowRes = filter * kspaceInput

    # Assuming kspaceInput and sz are defined, as in the previous piece of code
    kspaceInput = cmshiftnd(kspaceInput, [0, sz[0] // 2, sz[1] // 2, sz[2] // 2])
    kspLowRes = cmshiftnd(kspLowRes, [0, sz[0] // 2, sz[1] // 2, sz[2] // 2])

    # Reorder arrays for faster memory access
    #print(f"{kspaceInput.shape=}", f"{kspLowRes.shape=}") # Coil, kx, ky, kz
    # kspaceInput = np.moveaxis(kspaceInput, 0, -1) # move coil to last
    # kspLowRes = np.moveaxis(kspLowRes, 0, -1) # move coil to last
    # subs  = [*subs[1:],subs[0]]
    #print(subs)
    # Calculate initial image and the reference phase map
    fft_dims = (1,2,3)
    im = fftn(np.conj(kspaceInput), axes=fft_dims)  # The phase of Im might be wrong at this point
    phase = ifftn(kspLowRes, axes=fft_dims)
    phase = np.exp(1j * np.angle(phase))
    phase = phase / np.prod(sz)
    im = np.abs(im) * phase
    # print(f"{kspaceInput.flags=}")
    # print(f"{im.flags=}")
    # Assuming previous definitions of sz, subs, kspaceInput, and phase

    # In the loop, determine where to copy the measured data
    tmp = np.zeros(shape=(sz[pfDim],), dtype=bool)  
    tmp[:numSamples] = 1
    measured_idx = ifftshift(tmp)==1
    subs[pfDim+1] = np.nonzero(measured_idx)[0]
    kspaceInput = kspaceInput[tuple(subs)]
    # print("kspaceInput = kspaceInput[tuple(subs)]", subs)
    # Iteration loop
    for ii in range(iter):
        im = fftn(im, axes=fft_dims)        
        im[tuple(subs)] = kspaceInput
        im = np.conj(im)
        im = fftn(im, axes=fft_dims)
        im = np.abs(im) * phase



    # Calculate Ntrans as given
    Ntrans = (szSym-1) // 3
    #print(f"{Ntrans=}")
    # Create subscripts where we intend to keep the measured data, only.

    tmp = np.zeros(sz[pfDim], dtype=bool)
    tmp[:numSamples-Ntrans] = 1
    subsPure_idx = (ifftshift(tmp)==1)
    subsPure = subs.copy()
    subsPure[pfDim+1] = np.nonzero(subsPure_idx)[0]
    #print(f"{subsPure=}")
    # Create subscripts where we want to have a smooth transition between measured and phase-corrected data

    subsTrans_idx = subsPure_idx
    subsTrans_idx = np.logical_and(measured_idx, np.logical_not(subsPure_idx))
    subsTrans = subs.copy()
    subsTrans[pfDim+1] = np.nonzero(subsTrans_idx)[0]
    #print(f"{subsTrans=}")


    # Separate data in unfiltered part and transition zone
    tmp = np.zeros(im.shape, dtype=kspaceInput.dtype)
    tmp[tuple(subs)] = kspaceInput
    kspPure = tmp[tuple(subsPure)]
    kspTrans = tmp[tuple(subsTrans)]
    
    # Build a filter for the transition
   
    tmp = hanning_without0(2*Ntrans + 1)
    filterTrans = tmp[Ntrans+1:]
    filtershape = [1]*kspTrans.ndim
    filtershape[pfDim+1] = Ntrans
    filterTrans = np.reshape(filterTrans, filtershape)
    #print(f"{filterTrans.shape=}")



    # "im" becomes k-space signal, again
    im = fftn(im, axes=fft_dims)

    # Strict data consistency for numSamples-Ntrans samples
    im[tuple(subsPure)] = kspPure

    # Calculate transition using broadcasting
    #print(im[tuple(subsTrans)].shape,filterTrans.shape,kspTrans.shape)
    im[tuple(subsTrans)] = filterTrans * kspTrans + (1 - filterTrans) * im[tuple(subsTrans)]
    

    # If we are asked to return more than the image
    kspFull = im

    # Convert "im" back to image
    im = ifftn(im, axes=fft_dims)

    # Undo the prerequisites
    # Undo the permutations
    # im = np.moveaxis(im, -1, 0)
    # kspFull = np.moveaxis(kspFull, -1, 0)
    #print(f"{subs=}")
    # subs = [subs[-1], *subs[0:-1]]
    #print(f"{subs=}")
    # Undo the fftshifts
    # Assuming you have the cmshiftnd function translated into Python, here's how you'd call it
    im = cmshiftnd(im, [0, *np.array(sz) / 2])
    kspFull = cmshiftnd(kspFull, [0, *np.array(sz) / 2])

    # Undo flipping
    if isLower:
        subs[pfDim+1] = np.arange(sz[pfDim]-1, -1, -1)
        #print(f"line im = im[tuple(subs)]:{subs=}")
        im = im[tuple(subs)]
        kspFull = kspFull[tuple(subs)]

    # Check if the coil dimension was added
    if wasAddedCoilDim:
        Ny, Nx, *_ = im.shape
        im = np.reshape(im, (Ny, Nx, -1))
        kspFull = np.reshape(kspFull, (Ny, Nx, -1))
    
    return kspFull


# In[2]:


def detectPFdim(smplPtrn, wasAddedCoilDim):
    Ndim = smplPtrn.ndim
    sz = smplPtrn.shape

    pfDim = 0
    isUpper = False
    isLower = False
    isPartialFourier = [False] * Ndim

    for d in range(Ndim):
        subs = [np.ones(sz[d], dtype=int)] * Ndim
        subs[d] = np.arange(sz[d])

        idx_d = np.ravel_multi_index(subs, sz)
        oneCol = smplPtrn[tuple(subs)]

        reshRule = [1] * Ndim
        reshRule[d] = sz[d]
        oneCol = np.reshape(oneCol, reshRule)

        repRule = list(sz)
        repRule[d] = 1

        isPartialFourier[d] = np.array_equal(smplPtrn, np.tile(oneCol, repRule))

        if isPartialFourier[d]:
            pfDim = d
            numSamples = np.count_nonzero(oneCol)

            isUpper = np.array_equal(oneCol.flatten(), np.concatenate([np.ones(numSamples, dtype=bool), np.zeros(sz[d]-numSamples, dtype=bool)]))
            isLower = np.array_equal(oneCol.flatten(), np.concatenate([np.zeros(sz[d]-numSamples, dtype=bool), np.ones(numSamples, dtype=bool)]))

    if np.count_nonzero(isPartialFourier) == 0:
        raise ValueError('No partial Fourier dimension found.')
    elif np.count_nonzero(isPartialFourier) > 1:
        raise ValueError('Partial Fourier only allowed in 1 dimension!')

    return pfDim, isUpper, isLower, numSamples

def findSymSampled(ksp, pfDim, numSamples):
    Ndim = ksp.ndim - 1
    sz = ksp.shape[1:]

    tmp = np.squeeze(np.sum(np.abs(ksp), axis=0))
    for d in range(Ndim):
        if d != pfDim:
            tmp = np.max(tmp, axis=d)

    centreLine = np.argmax(tmp.flatten())

    startSym = centreLine - (numSamples - centreLine)
    endSym = centreLine + (numSamples - centreLine)
    idxSym = np.arange(startSym, endSym + 1)

    if np.any(idxSym < 0) or np.any(idxSym >= sz[pfDim]):
        centreLine = ksp.shape[pfDim] // 2
        startSym = centreLine - (numSamples - centreLine)
        endSym = centreLine + (numSamples - centreLine)
        idxSym = np.arange(startSym, endSym + 1)

    return centreLine, idxSym

def cmshiftnd(x, shifts):
    if not shifts or np.all(np.array(shifts) == 0):
        return x
    return np.roll(x, [int(shift) for shift in shifts], axis=tuple(range(len(shifts))))

def ndWindowFilter(sz, pfDim, szSym, idxSym):
    filter = np.ones(sz)
    Ndim = len(sz)

    for d in range(Ndim):
        if d != pfDim:
            filter_dim = np.hamming(sz[d])
        else:
            filter_dim = np.zeros(sz[d])
            filter_dim[idxSym] = hanning_without0(szSym)

        reshRule = [1] * Ndim
        reshRule[d] = sz[d]
        filter_dim = np.reshape(filter_dim, reshRule)

        filter *= filter_dim

    return filter

def hanning_without0(n):
    return hann(n+2)[1:-1]