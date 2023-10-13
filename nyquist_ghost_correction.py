"""
EPI Nyquist Ghost Correction in MRI
-----------------------------------

Magnetic Resonance Imaging (MRI) is a powerful medical imaging technique used for non-invasively 
visualizing internal structures of the body. Echo Planar Imaging (EPI) is a rapid imaging sequence 
in MRI, which however can suffer from artifacts known as Nyquist ghosts. These ghosts arise from 
inconsistencies between odd and even lines in k-space and can be influenced by various factors 
including hardware imperfections, patient motion, and eddy currents.

The package provided here is aimed at correcting Nyquist ghosts by employing a 1D linear phase model. 
The approach encompasses two distinct methods:

1. Entropy-Based Correction (`oneDimLinearCorr_entropy`): In this method, the k-space data undergoes 
an iterative optimization process using entropy as a cost function. Entropy-based methods aim to 
maximize the randomness or unpredictability of the corrected image. The correction parameters are 
initialized based on k-space data and further refined using an entropy-based optimization routine. 
This method automatically determines the phase correction parameters required for each slice.

2. Parameter-Based Correction (`oneDimLinearCorr_parameter`): If the phase correction parameters are 
already known or have been pre-calculated, this method allows for direct correction of Nyquist ghosts 
using the given parameters. This method is especially useful when correction parameters for certain 
slices are known a priori or are derived from a reference scan.

Underlying Mechanism
--------------------

The underlying concept revolves around manipulating the phase in k-space to correct the inconsistencies 
leading to Nyquist ghosts. The package incorporates:
- Phase Correction (`OIPhaseCor`): This function applies a given linear and constant phase correction 
to the k-space data.
- Entropy Calculation (`OGetEntropy`): Entropy serves as a cost function in the optimization process. 
The entropy of an image represents its randomness, and by maximizing this randomness, the appearance 
of structured artifacts (like Nyquist ghosts) can be minimized.

Usage
-----

Users provide raw k-space data as an input and specify the number of shots (for multi-shot EPI). 
Depending on whether they want to use the entropy-based method or provide their correction parameters, 
they would call either `oneDimLinearCorr_entropy` or `oneDimLinearCorr_parameter`.

Author Information
------------------

Dr. Mengye Lyu
Shenzhen Technology University
lvmengye@sztu.edu.cn

Based on MATLAB implementation by Victor Xie et.al. at HKU BISP Lab.
"""
import numpy as np
from scipy.optimize import minimize
try:
    from pyfftw.interfaces.numpy_fft import fft, ifft, fftshift
except ImportError:
    from scipy.fft import fft, ifft, fftshift
    
def oneDimLinearCorr_entropy(epi_kxkyzc_raw, nShot):
    org_size = epi_kxkyzc_raw.shape

    if len(epi_kxkyzc_raw.shape) < 4:
        epi_kxkyzc_raw = epi_kxkyzc_raw.reshape(epi_kxkyzc_raw.shape[0], epi_kxkyzc_raw.shape[1], 1, epi_kxkyzc_raw.shape[2])

    epi_kxkyzc_lpcCor = np.zeros_like(epi_kxkyzc_raw)
    phasepara = np.zeros((epi_kxkyzc_lpcCor.shape[2], 2))
    nSlice = epi_kxkyzc_lpcCor.shape[2]
    assert epi_kxkyzc_lpcCor.shape[1]%2==0, "number of phase encoding must be a even number"
    middleSliceIndex = nSlice // 2

    for iSlice in list(range(middleSliceIndex, nSlice)) + list(range(middleSliceIndex - 1, -1, -1)):
        print(iSlice)
        
        data_kyxc = fftshift(fft(np.transpose(epi_kxkyzc_raw[:, :, iSlice, :], (1, 0, 2)), axis=1), axes=1)

        if iSlice == middleSliceIndex:
            CorxKy_en, phasepara[iSlice, :] = OIEntropyBasedCor_forCompile(data_kyxc, nShot, None)
        elif iSlice > middleSliceIndex:
            CorxKy_en, phasepara[iSlice, :] = OIEntropyBasedCor_forCompile(data_kyxc, nShot, phasepara[iSlice - 1, :])
        else:
            CorxKy_en, phasepara[iSlice, :] = OIEntropyBasedCor_forCompile(data_kyxc, nShot, phasepara[iSlice + 1, :])

        epi_kxkyzc_lpcCor[:, :, iSlice, :] = np.transpose(ifft(fftshift(CorxKy_en, axes=1), axis=1), (1, 0, 2))

    epi_kxkyzc_lpcCor = epi_kxkyzc_lpcCor.reshape(org_size)

    return epi_kxkyzc_lpcCor, phasepara

def oneDimLinearCorr_parameter(epi_kxkyzc_raw, nShot, phasepara_zp):
    """
    This function corrects Nyquist ghosts with 1D linear phase model using
    input parameters "phasepara_zp", whose dimensions should be slice-parameters.
    """
    
    org_shape = epi_kxkyzc_raw.shape

    # Ensure the array has 4 dimensions
    if len(epi_kxkyzc_raw.shape) < 4:
        epi_kxkyzc_raw = epi_kxkyzc_raw.reshape(epi_kxkyzc_raw.shape[0], 
                                                  epi_kxkyzc_raw.shape[1], 
                                                  1, 
                                                  epi_kxkyzc_raw.shape[2])

    epi_kxkyz_c_lpcCor = np.zeros_like(epi_kxkyzc_raw)
    nSlice = epi_kxkyz_c_lpcCor.shape[2]

    for iSlice in range(nSlice):
        Kyxc = fftshift(fft(np.transpose(epi_kxkyzc_raw[:, :, iSlice, :], (1, 0, 2)), axis=1), axes=1)
        
        # Assuming OIPhaseCor is the Python version of the given MATLAB function
        CorxKy_en = OIPhaseCor(Kyxc, phasepara_zp[iSlice, :], nShot)
        epi_kxkyz_c_lpcCor[:, :, iSlice, :] = np.transpose(ifft(fftshift(CorxKy_en, axes=1), 
                                                                axis=1), 
                                                           (1, 0, 2))

    # Reshape the result to its original shape
    epi_kxkyz_c_lpcCor = epi_kxkyz_c_lpcCor.reshape(org_shape)

    return epi_kxkyz_c_lpcCor


def OIPhaseCor(data_kyxc, PhasPara, Nshot):
    ConPhase = PhasPara[0]
    LinPhase = PhasPara[1]

    Phase = np.reshape(ConPhase + np.arange(data_kyxc.shape[1]) * LinPhase, (1,-1, 1))
    Phasemap = np.tile(np.vstack((Phase, -1*Phase)), (data_kyxc.shape[0] // (2 * Nshot), 1, data_kyxc.shape[2]))
    Cor_data_kyxc = np.zeros_like(data_kyxc)
    
    for ii in range(Nshot):
        Cor_data_kyxc[ii::Nshot, :, :] = data_kyxc[ii::Nshot, :, :] * np.exp(1j * Phasemap)

    return Cor_data_kyxc

def OIEntropyBasedCor_forCompile(data_kyxc, Nshot, StartPoint=None):
    if StartPoint is None:
        StartPoint = IGetStartPoint(data_kyxc, [-np.pi/3, np.pi/3], [-0.1, 0.1], [50, 30], Nshot)

    res = minimize(lambda x: OGetEntropy(data_kyxc, x, Nshot), StartPoint, method='Nelder-Mead')
    PhasePara = res.x
    PhasePara[0] = (PhasePara[0] + np.pi/2) % np.pi - np.pi/2
    AfterCor = OIPhaseCor(data_kyxc, PhasePara, Nshot)

    # Additional computations here to emulate the MATLAB function
    # ...

    return AfterCor, PhasePara

def IGetStartPoint(data_kyxc, ConRange, LinRange, Nstep, Nshot):
    ConPhase = np.linspace(ConRange[0], ConRange[1], Nstep[0]+1)
    LinPhase = np.linspace(LinRange[0], LinRange[1], Nstep[1]+1)

    E = np.zeros((Nstep[0]+1, Nstep[1]+1))

    for i in range(Nstep[0]+1):
        for j in range(Nstep[1]+1):
            E[i, j] = OGetEntropy(data_kyxc, [ConPhase[i], LinPhase[j]], Nshot)

    r, c = np.unravel_index(np.argmin(E, axis=None), E.shape)
    Location = [ConPhase[r], LinPhase[c]]

    return Location

def OGetEntropy(data_kyxc, PhaPara=None, Nshot=None):
    if PhaPara is not None:
        x = PhaPara
    else:
        x = [0, 0]

    AfterCor = OIPhaseCor(data_kyxc, x, Nshot)

    # Restricting the image to the central k-space for resolution reduction
    image = fft(AfterCor[int(np.ceil(AfterCor.shape[0] / 4)):int(np.ceil(AfterCor.shape[0] * 3 / 4)),
                        int(np.ceil(AfterCor.shape[1] / 4)):int(np.ceil(AfterCor.shape[1] * 3 / 4)), :], axis=0)
    
    image = np.sum(np.abs(image)**2, axis=2)
    
    SquareSum = np.sum(image)
    B = np.sqrt(image) / SquareSum
    PointwiseEntropy = B / np.log(B)
    entropy = -np.sum(PointwiseEntropy)

    return entropy
