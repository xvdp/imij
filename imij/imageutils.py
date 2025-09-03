"""
"""
import os
import os.path as osp
import numpy as np
import cv2
from PIL import Image
from .utils import get_next_file_path


def histdif(x : np.ndarray, y : np.ndarray) -> float:
	""" compute histogram difference
    assumes x in np.uint8

	calcHist(images, -> in HSV
             channels, -> H, S
             mask - > None # could add face masks?
             histSize - > [ 50, 60 ], H: 50, s: 60
             ranges[, hist[, accumulate]]) -> [0, 180, 0, 256 ] H:[0-180], V[0-256]
    compareHist( methods:  Correlation      COMP_CORREL
                           Chi-Square       COMP_CHISQR 
                           Intersection     CCOMP_INTERSECT
                           Bhattacharyya    COMP_BHATTACHARYYA
    https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
	"""
	hist_x = cv2.calcHist([cv2.cvtColor(x, cv2.COLOR_BGR2HSV)], [ 0, 1 ], None, [ 50, 60 ], [ 0, 180, 0, 256 ])
	hist_y = cv2.calcHist([cv2.cvtColor(y, cv2.COLOR_BGR2HSV)], [ 0, 1 ], None, [ 50, 60 ], [ 0, 180, 0, 256 ])
	return float(np.interp(cv2.compareHist(hist_x, hist_y, cv2.HISTCMP_CORREL),
						   [ -1, 1 ], [ 0, 1 ]))

def map_np8_range(img, range=[0,256], b16=None):
    """ assumes float 32 is in range 0-1
        float16 may be 10, 12, 14, 16 bits
    """
    dtypes = (np.float32, np.float64, np.uint16, np.uint8)
    assert img.dtype in dtypes, f"expected dtypes {dtypes}, got {img.dtype}"
    if img.dtype in (np.float32, np.float64):
        return (range[0]/256, range[1]/256)
    if img.dtype == np.float16:
        if not b16:
            b16 = np.ceil(np.log2(img.max()))
            b16s = np.array([10,12,14,16])
            dif = b16s - b16
            difbits = 2**(dif[dif>=0][0] + b16 - 8)
        return range[0]*2**difbits, range[1]*2**difbits
    return range


# convert for ffmpeg: 
def to_uint16(image, bits=16):
    """ input float32 or 64 in range (0-1) approx
        output uint16 in range 0 - (2**bits -1)
    FFMPEG requires yuv...p10le, 12le, 14le, 16le as 16 bit for conversion
    """
    assert bits in (10,12,14,16)
    assert image.dtype in (np.float32, np.float64)
    _max =  2**bits-1 # max bits
    return np.clip((image * _max).round(),0, _max).astype(np.uint16)

def to_uint8(image):
    """ input float32 or 64 in range (0-1) approx
        output uint8 in range 255
    """
    if image.dtype == np.uint16:    
        image = image/(2**16 - 1)
    if image.dtype in (np.float32, np.float64):
        image = np.clip((image * 255).round(),0, 255).astype(np.uint8)
    assert image.dtype == np.uint8, f"did not convert image {image.dtype}"
    return image

def save_png(image, fname, channel_order='RGB', bits=16):
    """ saves 8bit or 16 bit pngs
    Args
        image   ndarray H,W,C or H,@  dtype np.uint8, np.uint16, np.float32, np.float64
                2 channels are graycale
                read with cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        fname
        channel_order   RGB BGR
        bits    optional if dtype is float 
    """
    try:
        assert isinstance(image, np.ndarray), f"expected ndarray, got {type(image)}"
        assert image.ndim in (2,3), f"expected H,W,C got {image.shape}"
        fname = osp.abspath(osp.expanduser(fname))
        folder = osp.dirname(osp.abspath(osp.expanduser(fname)))
        os.makedirs(folder, exist_ok=True)
        if image.ndim == 3 and channel_order=="RGB":
            image = image[:, :, ::-1]

        if image.dtype in (np.float32, np.float64):
            if bits == 8:
                image = to_uint8(image)
            else:
                image = to_uint16(image)
        assert image.dtype in (np.uint8, np.uint16), f"wrong dtype {image.dtype}"
        cv2.imwrite(fname, image)
    except:
        print(f"could not write image {fname}\n{image}")

def save_png_seq(image, fname, channel_order='RGB', bits=16):
    fname = get_next_file_path(fname)
    save_png(image, fname, channel_order=channel_order, bits=bits)


def load_png(fname, as_float=32, channel_order="RGB"):
    """ load png either as its own dtype or to float
    """
    image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
    if len(channel_order) == 3 and image.ndim == 3 and image.shape[-1] == 4:
        image = image[...,:3]
    if channel_order == "RGB":
        image = image[:,:,::-1]
    if as_float:
        denom = 2 ** (8 * image.dtype.itemsize) - 1
        dtype = np.float64 if as_float == 64 else np.float32
        image = image.astype(dtype) / denom
    return image