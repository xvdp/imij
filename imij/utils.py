from typing import Union, Optional
import sys
import os
import os.path as osp
from pathlib import Path
import numpy as np
import re

def find_files(path: Optional[str] = None,
               extensions: tuple = ('.ccc', '.cube'),
               max_count: int  = 1,
               search_dirs = ('.', '..')):
    """
    Recursively searches for files with .ccc and .cube extensions
    in the current directory (.) and parent directory (..).
    
    Returns:
        dict: A dictionary with keys 'ccc_files' and 'cube_files' 
              containing lists of found file paths.
    """

    search_dirs = ['.', '..']
    if path not in (None, ".", ""):
        search_dirs = [osp.join(path, s) for s in search_dirs]

    out = {ext: [] for ext in extensions}

    for search_dir in search_dirs:
        search_path = Path(search_dir).resolve()
        
        for root, _, files in os.walk(search_path):
            for file in files:
                for ext in out:
                    if len(out[ext]) < max_count and file.endswith(ext):
                         out[ext].append(str(Path(root) / file))
                if all([len(it)>=max_count for it in out.values()]):
                    return out
    return out


def get_next_file_path(filename):
    folder_path = osp.expanduser(osp.abspath(osp.dirname(filename)))
    filename = osp.basename(filename)
    os.makedirs(folder_path, exist_ok=True)
    ext = "."+filename.split(".")[-1]
    fname = filename[:-(len(ext))]
    
    # Get existing files to determine next index
    existing_files = [f for f in os.listdir(folder_path)if f.startswith(fname) and f.endswith(ext)]
    
    # Extract numbers from existing filenames
    indices = []
    for _fname in existing_files:
        match = re.search(rf"{fname}_(\d+)\{ext}", _fname)
        if match:
            indices.append(int(match.group(1)))
    
    # Determine next index
    next_index = max(indices) + 1 if indices else 1
    
    # Create filename
    filename = f"{fname}_{next_index:03d}{ext}"
    return osp.join(folder_path, filename)
