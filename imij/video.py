import os
import subprocess as sp
from .imageutils import to_uint8, to_uint16
from .dpx import opendpx

# wip wip wip
def crop_and_video(files, crops, processor, cmd, bits=10 ):
    """ wip dpx files to cropped video, requires dpx pprocessor and ffmpeg command
    files   dpx file sequence
    crops   tuple of slices (slice(0, None), slice(0, None)) doesnt crop
    bits    if cmd includes prores yuv10le then 2 10 bits.
    """
    if isinstance(files, str):
        files = [f.path for f in os.scandir(files) if f.is_file() and f.name.endswith('.dpx')]
        files.sort()
    print(cmd)
    
    sp = sp.Popen(cmd, stdin=sp.PIPE)
    for i, file in enumerate(files):
        print(f"{i+1}|{len(files)}\t")
        image = opendpx(file, processor)[crops[0], crops[1]]
        if bits == 8:
            image = to_uint8(image)
        else:
            image = to_uint16(image, 16)
        sp.stdin.write(image.tobytes()) 
    sp.stdin.close()
    sp.wait()
    print("\ndone")