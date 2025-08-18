

from setuptools import setup
import os
import sys
from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement

def _set_version(version):
    with open('imij/version.py', 'w', encoding='utf8') as _fi:
        _fi.write("version='"+version+"'")
        return version

def is_conda_environment():
    return os.getenv("CONDA_PREFIX") is not None

def get_installed_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None

def check_for_conflicts(requirements):
    conflicts = []
    for req_str in requirements:
        req = Requirement(req_str)
        installed_version = get_installed_version(req.name)
        if installed_version and not req.specifier.contains(installed_version):
            conflicts.append((req.name, installed_version, str(req.specifier)))
    return conflicts

INSTALL_REQUIRES = ["numpy<=1.26.4",
                    "python<=3.12.11"] 
INSTALL_REQUIRES_PIP = ["opencv-python<=4.11.0.86",
                        "opencv-contrib-python<=4.11.0.86",
                        "opencolorio",
                        "openimageio",
                        "cdl_convert"] 

if is_conda_environment():
    conflicts = check_for_conflicts(INSTALL_REQUIRES)
    if conflicts:
        print("\n WARNING: Potential package conflicts in Conda environment!")
        for pkg, ver, spec in conflicts:
            print(f"  - {pkg} (installed: {ver}, required: {spec})")
        print("\nRecommendation: Install dependencies via Conda/Mamba first:")
        print(f"  mamba install {' '.join(Requirement(req).name for req in INSTALL_REQUIRES)}")
        if input("Continue with pip install? (y/n): ").lower() != 'y':
            sys.exit(1)
else:
    INSTALL_REQUIRES_PIP += INSTALL_REQUIRES


setup(
    name="imij",
    install_requires=INSTALL_REQUIRES_PIP,
    url='http://github.com/xvdp/imij',
    version=_set_version(version='0.0.1')
)