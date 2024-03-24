# pymrghost
python implementation of MRI EPI ghost correction, partial Fourier reconstruction, and other common functions.

# Usage
 - Pass 2d multislice epi k-space data with dimensions of [Kx, Ky, Slice, Coil] to the function `nyquist_ghost_correction.oneDimLinearCorr_entropy` for entropy-based referenceless Nyquist ghost correction. (may not work well for accelerated data though)
 - Pass 2d multislice k-space data with dimensions of [Kx, Ky, Slice, Coil] to the function `partial_fourier_recon.pf_recon_pocs_ms2d` for pocs-based partial Fourier reconstruction. (Kx should be the axis where partial Fourier is applied, otherwise specify `pf_dim` as input parameter)
# You may find the following papers useful:
- Skare S, Clayton D, Newbould R, Moseley M, Bammer R. A fast and robust minimum entropy based non-interactive Nyquist ghost correction algorithm. InProceedings of the 14th Annual Meeting of ISMRM 2006 (p. 460). https://archive.ismrm.org/2006/2349.html
- Xie VB, Lyu M, Liu Y, Feng Y, Wu EX. Robust EPI Nyquist ghost removal by incorporating phase error correction with sensitivity encoding (PEC‐SENSE). Magnetic resonance in medicine. 2018 Feb;79(2):943-51. https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26710
- Lyu M, Barth M, Xie VB, Liu Y, Ma X, Feng Y, Wu EX. Robust SENSE reconstruction of simultaneous multislice EPI with low‐rank enhanced coil sensitivity calibration and slice‐dependent 2D Nyquist ghost correction. Magnetic Resonance in Medicine. 2018 Oct;80(4):1376-90. https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27120
- Liu Y, Lyu M, Barth M, Yi Z, Leong AT, Chen F, Feng Y, Wu EX. PEC‐GRAPPA reconstruction of simultaneous multislice EPI with slice‐dependent 2D Nyquist ghost correction. Magnetic Resonance in Medicine. 2019 Mar;81(3):1924-34. https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.27546
