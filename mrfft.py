__author__ = 'Jo Schlemper;Mengye Lyu'

# from tensorflow.signal import fft, fft2d, ifft2d, ifft, ifftshift, fftshift
from numpy.fft import fft, fft2 as fft2d, ifft2 as ifft2d, ifft, ifftshift, fftshift
import numpy as np

def fftc(x, axis=- 1):
    ''' expect x as m*n matrix '''
    return fftshift(fft(ifftshift(x), axis=axis, norm="ortho"))


def ifftc(x, axis=- 1):
    ''' expect x as m*n matrix '''
    return fftshift(ifft(ifftshift(x), axis=axis, norm="ortho"))


def fft2c(x, axes=(- 2, - 1)):
    '''
    Centered fft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    # axes = (len(x.shape)-2, len(x.shape)-1)  # get last 2 axes
    #axes = (-2, -1)  # get last 2 axes
    res = fftshift(fft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res


def ifft2c(x, axes=(- 2, - 1)):
    '''
    Centered ifft
    Note: fft2 applies fft to last 2 axes by default
    :param x: 2D onwards. e.g: if its 3d, x.shape = (n,row,col). 4d:x.shape = (n,slice,row,col)
    :return:
    '''
    #axes = (-2, -1)  # get last 2 axes
    res = fftshift(ifft2d(ifftshift(x), axes=axes, norm="ortho"))
    return res

def sos(x, axis=- 1):
    '''
    root mean sum of squares, default on first dim
    '''
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res
    
def rsos(x, axis=0):
    '''
    root mean sum of squares, default on first dim
    '''
    res = np.sqrt(np.sum(np.abs(x)**2, axis=axis))
    return res

def zpad(array_in, outshape):
    import math
    #out = np.zeros(outshape, dtype=array_in.dtype)
    oldshape = array_in.shape
    assert len(oldshape)==len(outshape)
    #kspdata = np.array(kspdata)
    pad_list=[]
    for iold, iout in zip(oldshape, outshape):
        left = math.floor((iout-iold)/2)
        right = math.ceil((iout-iold)/2)
        pad_list.append((left, right))

    zfill = np.pad(array_in, pad_list, 'constant')                     # fill blade into square with 0
    return zfill

def crop(img, bounding):
    import operator
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices].copy()