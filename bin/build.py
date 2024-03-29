import sys
import os
from os import path as osp
from cx_Freeze import setup, Executable

my_dir = osp.dirname(osp.abspath(__file__))
os.chdir(my_dir)

company_name = "JRC"
product_name = "co2mpas_driver"

bdist_msi_options = {
    "upgrade_code": "",
    "add_to_path": False,
    "initial_target_dir": r"[ProgramFilesFolder]\%s\%s" % (company_name, product_name),
}

path = sys.path
build_exe_options = {"path": path, "icon": "brain.ico"}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

exe = Executable(
    script="co2mpas_driver/examples/sample_4degree_with_linear.py",
    base=base,
    icon=None,
)

setup(
    name="co2mpas_driver",
    version="1.0.0",
    description="A lightweight microsimulation free-flow acceleration model"
    "(MFC) that is able to capture the vehicle acceleration"
    "dynamics accurately and consistently",
    executables=[exe],
    options={"bdist_msi": bdist_msi_options},
)

# Please run this script with a second file
# py build.py build
# py build.py bdist_msi
