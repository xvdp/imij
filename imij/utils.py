from typing import Union, Optional
import sys
import os
import os.path as osp
from pathlib import Path

def find_files(path: Optional[str] = None,
               extensions: tuple = ('.ccc', '.cube'),
               max_count: int  = 1):
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

