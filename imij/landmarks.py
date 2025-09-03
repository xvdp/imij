import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_next_file_path


def save_np(fname, landmarks):
    np.save(fname, landmarks)

def save_np_seq(fname, landmarks):
    fname = get_next_file_path(fname)
    np.save(fname, landmarks)


def load_np(fname, plot=False, show=False, **kwargs):
    """
    fname     str type .npz  shape [n, 2]
    """
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")
    landmarks = np.load(fname)
    if plot:
        title = kwargs.pop('title', None)
        if title is not None:
            plt.title(title)
        plt.scatter(landmarks.T, **kwargs)
        if show:
            plt.show()
    return landmarks


def draw_crop(x,dx,y,dy, **kwargs):
    """ plot crop location info
    """
    text = kwargs.pop("text", None)
    text_loc = kwargs.pop("text_loc", (0,2))
    fontsize = kwargs.pop("fontsize", None)
    color = kwargs.pop("color", "yellow")
    corners = kwargs.pop('corners', False)
    offset = kwargs.pop('offset', (0,0))
    rotation = kwargs.pop('rotation', 0)
    image_size = kwargs.pop('image_size', None)
    xs = [x, x+dx, x+dx, x, x]
    ys = [y, y, y+dy, y+dy, y]
    plt.plot(xs, ys, color=color, **kwargs)
    if text is not None:
        plt.text(x + dx*text_loc[0], y + dy*text_loc[1], s=text, color=color, fontsize=fontsize)
    if corners:
        for i in range(4):
            plt.text(xs[i]+ offset[0], ys[i]+offset[1], s=f"{xs[i], ys[i]}", color=color, fontsize=fontsize, rotation=rotation)
        plt.text(0,0, s=f"{0, 0}", color=color, fontsize=fontsize, rotation=rotation, ha='left', va='top')
        if image_size is not None:
            sx, sy = image_size[:2]
            plt.text(sx, sy, s=f"{sx, sy}", color=color, fontsize=fontsize, rotation=rotation, ha='right', va='bottom')
        
