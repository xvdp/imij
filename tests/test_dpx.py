import pytest
import numpy as np
import PyOpenColorIO as ocio
from unittest.mock import patch, MagicMock

import imij.dpx as dpx

class DummyProcessor:
    def apply(self, img_desc):
        # Simulate a processor that does nothing
        pass

class DummyImageDesc:
    def __init__(self, data, W, H, C):
        self._data = data
        self._shape = (H, W, C)
    def getData(self):
        return self._data.reshape(self._shape)

# replace PackedImageDesc class ocio with DummyImageDesc class during the test.
@patch("imij.dpx.ocio.PackedImageDesc", DummyImageDesc)
def test_dpxcpuproc_shape():
    arr = np.ones((4, 5, 3), dtype=np.float32)
    proc = DummyProcessor()
    out = dpx.dpxcpuproc(arr, proc)
    assert out.shape == (4, 5, 3)

def test_cdl_from_ccc_none():
    cdl = dpx.cdl_from_ccc(None)
    assert np.allclose(cdl.getSlope(), [1.0, 1.0, 1.0])
    assert np.allclose(cdl.getOffset(), [0.0, 0.0, 0.0])
    assert np.allclose(cdl.getPower(), [1.0, 1.0, 1.0])
    assert cdl.getSat() == 1


def test_invert_cdl_real():
    cdl = ocio.CDLTransform()
    cdl.setSlope([1.0, 2.0, 0.5])
    cdl.setOffset([0.1, -0.2, 0.3])
    cdl.setPower([2.0, 2.0, 2.0])
    cdl.setSat(2.0)
    inv = dpx.invert_cdl(cdl)
    assert isinstance(inv, ocio.CDLTransform)

def test_DP_class_make_processor(monkeypatch):
    monkeypatch.setattr(dpx, "make_image_processor", lambda lut, ccc: "PROC")
    D = dpx.DPX()
    D.make_processor("lut.cube", "ccc.ccc")
    assert D.proc == "PROC"

def test_DP_class_open(monkeypatch):
    D = dpx.DPX()
    monkeypatch.setattr(dpx, "opendpx", lambda fname, proc: "PIXELS")
    monkeypatch.setattr(dpx, "make_image_processor", lambda lut, ccc: "PROC")
    monkeypatch.setattr(dpx.osp, "isfile", lambda f: True)
    out = D.open("file.dpx", applylut=True, lut="lut.cube", ccc="ccc.ccc")
    assert out == "PIXELS"

def test_make_image_processor_no_lut(monkeypatch):
    monkeypatch.setattr(dpx, "image_proc_no_lut", lambda cdl: "NO_LUT_PROC")
    proc = dpx.make_image_processor(None, None)
    assert proc == "NO_LUT_PROC"