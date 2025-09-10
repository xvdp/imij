"""
"""
from typing import Optional, Union
import os
import os.path as osp
import math
import numpy as np
import cv2

from .utils import get_next_file_path

def _img_range(x: np.ndarray, bits: Optional[int] = None):
    """ return image range floats -> (0,1) or (x.min(), x.max())
    bits force min max for uint bits, eg. bits=10, range-> 0, 2**10
    """
    if bits is not None: # ignore image and force, assume image is within that range
        return 0, 2**bits - 1
    _dtypes = (np.float32, np.float64, np.uint8, np.uint16)
    assert x.dtype in _dtypes, f"Unhandled dtype {x.dtype}, expected {_dtypes}"
    if x.dtype in (np.float32, np.float64):
        out = min(x.min(), 0.0), max(x.max(), 1.0)
    else:
        out = (0, 2 ** (8 * x.dtype.itemsize) - 1)
    return out

def histdif(x : np.ndarray, y : np.ndarray, **kwargs) -> float:
    """ compute histogram difference in HSV over HS channels

    calcHist(images, -> in HSV
             channels, -> H, S
             mask - > None # could add face masks?
             histSize - > [ 50, 60 ], H: 50, sW: 60
             ranges, hist[, accumulate]]) -> [0, 180, 0, 256 ] H:[0-180], V[0-256]
    kwargs:
        hrange  tuple, [0, sin(pi/4)*img_range]
        vrange  tuple, [0, img_range]
        hsize   int [50]    bins for hue range
        vsize   int [60]    bins for value range
        method   [cv2.HISTCMP_CORREL] | cv2.COMP_CHISQR | cv2.CCOMP_INTERSECT | cv2.COMP_BHATTACHARYYA
    https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    """
    assert x.dtype == y.dtype, f"type mismatch, {x.dtype} != {y.dtype}"

    _, xmax = _img_range(x)
    hrange = kwargs.get('hrange', (0, (xmax*np.sin(np.pi/4).astype(x.dtype))))
    vrange = kwargs.get('vrange', (0, xmax))
    hsize =  kwargs.get('hsize', 50)
    vsize =  kwargs.get('vsize', 60)
    method = kwargs.get('method', cv2.HISTCMP_CORREL)
    hists = [cv2.calcHist([cv2.cvtColor(z, cv2.COLOR_BGR2HSV)],
                          [ 0, 1 ], None, [ hsize, vsize ], [ *hrange, *vrange ])
             for z in (x, y)]
    return float(np.interp(cv2.compareHist(hists[0], hists[1], method), [ -1, 1 ], [ 0, 1 ]))

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


# def load_png(fname, as_float=32, channel_order="RGB"):
#     """ load png either as its own dtype or to float
#     """
#     image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#     if len(channel_order) == 3 and image.ndim == 3 and image.shape[-1] == 4:
#         image = image[...,:3]
#     if channel_order == "RGB":
#         image = image[:,:,::-1]
#     if as_float:
#         denom = 2 ** (8 * image.dtype.itemsize) - 1
#         dtype = np.float64 if as_float == 64 else np.float32
#         image = image.astype(dtype) / denom
#     return image


## another load imagev


def load_png(name,
             dtype: Optional[str] = None,
             channel_order: Optional[str] = None,
             order: Optional[str] = None ):
    """ load png either as its own dtype or to float
        dtype:  None: unchanged
                'f32', 'f64': float
                'uint8':
        order   None -> as read
                HW3 -> ensure last channel is 3
                N3HW -> to 
                """
    kw = {} if dtype == 'uint8' else {cv2.IMREAD_UNCHANGED}
    img = cv2.imread(name, *kw)

    if order is None and channel_order is not None:
        order = f'HW{len(channel_order)}'

    if order is not None:
        _orders = ['HWC', 'HW3', 'N3HW', 'NCHW']
        if len(img.shape) == 3 and img.shape[2] == 1 and order not in ('HCW', 'NCHW'):
            img = img[:,:,0]
            
        if len(img.shape) == 2:
            _repeats = 1
            if order in ('HW3', 'N3HW'):
                _repeats = 3
            elif order in  ('HW4', 'N4HW'):
                _repeats = 4
            img = np.stack([img]*_repeats, axis=-1)
        else:
            if order in ('HW3', 'N3HW') and img.shape[2] == 4:
                img = img[..., :3]
            elif order in ('HW4', 'N4HW') and img.shape[2] == 3:
                img = np.concatenate([img, np.zeros_like(img[:,:,0:1])], axis=-1)
    if channel_order is not None:
        if channel_order == "RGB":
            img = img[:,:,::-1]
    if dtype in ('f32', 'f64'):
        _dtype = {'f32':np.float32, 'f64':np.float32}[dtype]
        img = img.astype(_dtype) / _img_range(img)
    if dtype == 'uint16' and img.dtype == np.uint8:
        img = img.astype(np.uint16)*256
    return img


def load_scaled_image(name: str,
                      to_range: tuple = (0,1),
                      size: Union[np.ndarray, tuple, None] = (640, 640), # h, w
                      channel_order: Optional[str] = None,
                      dtype: str = 'f32') -> tuple: # shape [1, 3, H, W], scale
    """ to_range        opened in range 0,1 by default
        size h,w
    """
    img = load_png(name, dtype=dtype, order='HW3', channel_order=channel_order)
    _imsize = np.array(img.shape[:2])
    rescale = np.ones(2, dtype=img.dtype)
    if to_range != (0,1):
        img * (to_range[1] - to_range[0]) + to_range[0]
    if size is not None:
        img = resize_fit(img, size)
        rescale = _imsize/np.array(size)
        img = expand_black(img, size)

    return img.transpose(2,0,1)[None].astype(np.float32), rescale



def resize_fit(img : np.ndarray, size : Union[np.ndarray, tuple], **kwargs) -> np.ndarray:
    """resize to fit 
        size h,w
    """
    h, w = img.shape[:2]
    if (h,w) != size:
        scale = min(size[0]/h, size[1]/w)
        _w = int(w * scale)
        _h = int(h * scale)
        return cv2.resize(img, (_w, _h), **kwargs)
    return img

def expand_to_mult(img: Union[np.ndarray, str],
                   mult: int,
                   color: Union[int, float] = 0,
                   out_name: Optional[str] = None) -> np.ndarray:
    """  some models require images that are divisible by 8 for example
    """
    if isinstance(img, str):
        out_name = img if out_name is None else out_name
        img = load_png(img)
    shape = list(img.shape)
    h,w = shape[:2]
    hout = math.ceil(h/mult)*mult
    wout = math.ceil(w/mult)*mult
    if hout != h or wout != w:
        shape[0] = hout
        shape[1] = wout
        img = expand_image(img, tuple(shape), color=color)
        if out_name:
            cv2.imwrite(out_name, img)
    return img

def crop_to_mult(img: Union[np.ndarray, str],
                   mult: int,
                   color: Union[int, float] = 0,
                   out_name: Optional[str] = None) -> np.ndarray:
    """  some models require images that are divisible by 8 for example
    """
    if isinstance(img, str):
        out_name = img if out_name is None else out_name
        img = load_png(img)
    shape = list(img.shape)
    h,w = shape[:2]
    hout = math.floor(h/mult)*mult
    wout = math.floor(w/mult)*mult
    if hout != h or wout != w:
        dh = (h - hout)
        dw = (w - wout)
        img = img[dh//2:-(dh - dh//2), dw//2:-(dw - dw//2)]

        if out_name:
            cv2.imwrite(out_name, img)
    return img

def expand_black(img : np.ndarray, size: Union[tuple, np.ndarray]) -> np.ndarray:
    """ paste img in black background centered
        to float float32 / in range 0,255, H,W,C -> N,H,W,C
        size h,w
    """
    return expand_image(img, size, 0)

def expand_image(img : np.ndarray, size: Union[tuple, np.ndarray], color=0) -> np.ndarray:
    """ paste img in black background centered
        to float float32 / in range 0,255, H,W,C -> N,H,W,C
        size h,w
    """
    if color == 1 and img.dtype not in (np.float32, np.float64):
        color = (2 ** (8 * img.dtype.itemsize) - 1)
    h, w, c = img.shape[:3]
    dh = size[0] - h
    dw = size[1] - w
    if dh or dw:
        hslice = slice(0, None) if dh == 0 else slice(dh//2,  dh//2 + h)
        wslice = slice(0, None) if dw == 0 else slice(dw//2,  dw//2 + w)
        out_img = np.ones((size[0], size[1], c), dtype=img.dtype)*color
        out_img[hslice, wslice, :] = img
    else:
        out_img = img
    return out_img

def norm_to_range(img : np.ndarray, norm_range = (-1, 1)) -> np.ndarray:
    if norm_range ==  (-1, 1):
        return (img - 127.5) / 128.0
    if norm_range == ( 0, 1 ):
        return img / 255.0
    return img
