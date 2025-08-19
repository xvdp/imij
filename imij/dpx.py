import os
import os.path as osp
from typing import Union, Optional
from copy import deepcopy
import numpy as np
import OpenImageIO as oiio
import PyOpenColorIO as ocio
import cdl_convert.parse as cdl_parse

from .utils import find_files

class DPX:
    """
    shortcut for mucking around with dpx files,  

    example:
        >>> D = DPX()
        >>> image = D.open("dpxfile.dpx", applylut=True)
        # if theres a lut recursively, from ../ it will apply the lut and the color correction
        # otherwise
        >>> D.make_processor(lut, ccc)
        >>> D.open("dpxfile.dpx", applylut=True)
    """

    def __init__(self):
        self.proc = None

    def make_processor(self,
                       lut:Union[str, ocio.FileTransform, None] = None,
                       ccc:Union[str, ocio.CDLTransform, None] = None):
        self.proc = make_image_processor(lut, ccc)

    def apply_ccc(self, image: np.ndarray):
        assert self.proc is not None,\
            f"missing processor: > .make_processor(lut, ccc)"
        return dpxcpuproc(image)

    def open(self, dpxfile: str, applylut: bool = True, **kwargs):
        """ searches for ccc and cube in the neighborhood of dpx file if no processor present
        """
        assert osp.isfile(dpxfile), f"dpx file not found in path {dpxfile}"
        lut = kwargs.pop('lut', None)
        ccc = kwargs.pop('ccc', None)
        assert lut is None or osp.isfile(lut), f" lut kwarg expects file got {type(lut)}" 
        assert ccc is None or osp.isfile(ccc), f" ccc kwarg expects file got {type(ccc)}" 
        if lut is not None or ccc is not None:
            self.make_processor(lut, ccc)
            applylut = True
  
        if applylut:
            if self.proc is None:
                lut, ccc = self._get_lut_ccc(dpxfile)
                if ccc is None and lut is None:
                    print("No ccc or lut files were found, opening raw")
                else:
                    print (f" applying lut {lut}, and ccc {ccc}")
                self.make_processor(lut, ccc)
        processor = self.proc if applylut else None
        return opendpx(dpxfile, processor)

    @staticmethod
    def _get_lut_ccc(path="."):
        if osp.isfile(path):
            path = osp.dirname(path)
        files = find_files(osp.dirname(path), extensions=(".ccc", ".cube"))
        ccc = files[".ccc"][0] if files[".ccc"] else None
        lut = files[".cube"][0] if files[".cube"] else None
        return lut, ccc


def opendpx_ccc(fname: str,
                lut:Union[str, ocio.FileTransform, None] = None,
                cdl:Union[str, ocio.CDLTransform, None] = None):
    """ opens dpx file, creates color correction
    """
    if (lut or cdl):
        processor = make_image_processor(lut, cdl)
    else:
        processor = None
    return opendpx(fname, processor)


def dpxcpuproc(image, cpuproc):
    H, W, C = image.shape
    flat = np.ascontiguousarray(image.reshape(-1, C), dtype=np.float32)
    img_desc = ocio.PackedImageDesc(flat, W, H, C)
    cpuproc.apply(img_desc)
    return img_desc.getData().reshape(H, W, C)


def opendpx(fname, proccessor=None):
    """ opens dpx file
        if processor to apply lut and cdl is passed then opens with
    """
    buf = oiio.ImageInput.open(fname)
    spec = buf.spec()
    pixels = buf.read_image(oiio.FLOAT)
    buf.close()
    if proccessor is not None:
        print("applying proc")
        pixels = dpxcpuproc(pixels, proccessor)
    return pixels

def cdl_from_ccc(cccfile=None):
    """ cccfile : opens filename.cc color correction file and applies the t transform 
    """
    if cccfile is not None:
        cc = cdl_parse.parse_ccc(cccfile) 
        slope  = cc.all_children[0].slope
        offset  = cc.all_children[0].offset
        power  = cc.all_children[0].power
        sat  = cc.all_children[0].sat
    else:
        print ("No cdl")
        slope = [1.0, 1.0, 1.0]
        offset = [0.0, 0.0, 0.0]
        power = [1.0, 1.0, 1.0]
        sat = 1

    cdl = ocio.CDLTransform()
    cdl.setSlope(slope)
    cdl.setOffset(offset)
    cdl.setPower(power)
    cdl.setSat(sat)
    return cdl


def invert_cdl(cdl):
    """ reads cdl and inverts the correction slopes
    """
    if (isinstance(cdl, str) and osp.isfile(cdl)):
        assert cdl.endswith(".ccc"), f"exptcted file .ccc, got {cdl}"
        cdl = cdl_from_ccc(cdl)
    assert isinstance(cdl, ocio.CDLTransform),\
        f"incorrect cdl type, either file.ccc or CLDTransform, got {type(cdl)}"

    inv_cdl = ocio.CDLTransform()
    invslope =  [1/s for s in cdl.getSlope()]
    invoffset = [-o for o in cdl.getOffset()]
    for i, o in enumerate(invoffset):
        invoffset[i] *= invslope[i]
    inv_cdl.setSlope(invslope)
    inv_cdl.setOffset(invoffset)
    inv_cdl.setPower([1/p for p in cdl.getPower()])
    inv_cdl.setSat(1/cdl.getSat())
    return inv_cdl

def invert_cdl2(cdl):
    """ WIP - theres a direction inverse. neither inversions work
    """
    if (isinstance(cdl, str) and osp.isfile(cdl)):
        inv_cdl = cdl_from_ccc(cdl)
    else:
        inv_cdl = deepcopy(cdl) 
    assert isinstance(inv_cdl, ocio.CDLTransform)
    inv_cdl.setDirection(ocio.TransformDirection.TRANSFORM_DIR_INVERSE)
    return (inv_cdl)
    

def readlut(lutfile):
    """ reads lut file """
    assert osp.isfile(lutfile) and lutfile.endswith(".cube"), \
        f"no lut.cube file not found {lutfile}"
    return ocio.FileTransform(src=lutfile, interpolation=ocio.INTERP_TETRAHEDRAL)

def make_image_processor(lut:Union[str, ocio.FileTransform, None] = None,
                         cdl:Union[str, ocio.CDLTransform, None] = None):
    """opencolorio supports gl processors, but those require writing a shader,
    we always use device="cpu"
    Args
        lut     .cube   lookup table, file or FileTransform
        cdl     .ccc    color correction file or CDLTransform
    """
    if lut is None:
        return image_proc_no_lut(cdl)
    
    if isinstance(lut, str) and osp.isfile(lut):
        lut = readlut(lut)
    if (isinstance(cdl, str) and osp.isfile(cdl)) or cdl is None:
        cdl = cdl_from_ccc(cdl)
    
    assert isinstance(lut, ocio.FileTransform), f"lut {type(lut)}"
    assert isinstance(cdl, ocio.CDLTransform), f"cdl {type(cdl)}"
    
    cfg = ocio.Config()
    film = ocio.ColorSpace(name='FilmLog')
    film.setFamily('Input')
    film.setDescription('Original film-log DPX')
    cfg.addColorSpace(film)
    grp = ocio.GroupTransform([cdl, lut])

    disp = ocio.ColorSpace(name='Display')
    disp.setFamily('Display')
    disp.setTransform(grp, ocio.COLORSPACE_DIR_FROM_REFERENCE)  # FilmLog ? Display
    cfg.addColorSpace(disp)

    proc = cfg.getProcessor('FilmLog', 'Display')
    # proc = cfg.getProcessor(film, disp) # equivalent
    # context = ocio.Context() 
    # proc = cfg.getProcessor(context, 'FilmLog', 'Display')
    return proc.getDefaultCPUProcessor()


def image_proc_no_lut(cdl):
    """ wip
    """
    if (isinstance(cdl, str) and osp.isfile(cdl)) or cdl is None:
        cdl = cdl_from_ccc(cdl)
    assert isinstance(cdl, ocio.CDLTransform)
    cfg = ocio.Config()
    
    inv_cdl_space = ocio.ColorSpace(name='inv_cdl')
    inv_cdl_space.setTransform(cdl, ocio.COLORSPACE_DIR_FROM_REFERENCE)
    cfg.addColorSpace(inv_cdl_space)

    no_cdl = cdl_from_ccc(None)
    cdl_space = ocio.ColorSpace(name='cdl')
    cdl_space.setTransform(no_cdl, ocio.COLORSPACE_DIR_FROM_REFERENCE)
    cfg.addColorSpace(cdl_space)


    # Create a processor that applies only the inverse CDL transformation
    proc = cfg.getProcessor('cdl', 'inv_cdl')  # source and destination are same if only applying inverse CDL
    return proc.getDefaultCPUProcessor()
