import sys
import os
from os import path as osp
from cx_Freeze import setup, Executable

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)

company_name = "JRC"
product_name = "new_MFC"

bdist_msi_options = {
    "add_to_path": False,
    "initial_target_dir": "C:/Apps",
}

path = sys.path
build_exe_options = {"path": path, "icon": "brain.ico"}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

exe = Executable(
    script="new_MFC/examples/sample_4degree_with_linear.py",
    base=base,
    icon=None,
)

setup(
    name="new_MFC",
    version="1.0.0",
    description="A lightweight microsimulation free-flow acceleration model"
    "(MFC) that is able to capture the vehicle acceleration"
    "dynamics accurately and consistently",
    executables=[exe],
    options={"bdist_msi": bdist_msi_options},
)

# Please run build.bat file to generate an MSI installable file
