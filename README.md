## Com2pas_driver: Try it live
<!--move them to CONTRIBUTING.md -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JRCSTU/co2mpas_driver/master?urlpath=lab/tree/examples)

Access this Binder at the following URL:

https://mybinder.org/v2/gh/JRCSTU/co2mpas_driver/master

Click the binder badge to try it live without installing anything. 
This will take you directly to JupyterLab where we used Jupyter notebook to 
present examples on how to use co2mpas_driver model (i.e., MFC) to simulate 
the driver behaviour of a vehicle.

## What is co2mpas_driver?

Co2mpas_driver is a library used to implement a lightweight microsimulation 
free-flow acceleration model (MFC) that is able to capture the vehicle acceleration 
dynamics accurately and consistently, it provides a link between the model and 
the driver and can be easily implemented and tested without raising the 
computational complexity. The proposed model has been developed by the Joint Research Centre of the 
European Commission for more details https://journals.sagepub.com/doi/10.1177/0361198119838515

## Installation

1. **Download or clone co2mpas_driver**
    If you have access to the project under JRCSTU github repository then clone
    the co2mpas_driver project to your local machine.
   
        git clone https://github.com/JRCSTU/co2mpas_driver.git
       
2. **Install on your local machine**
    You can install on your machine using:
    
        pip install dist/co2mpas_driver-1.0.0-py2.py3-none-any.whl 
        
3. **In order to use co2mpas_driver library**
   you can start importing and using co2mpas driver on python console.
   
        import co2mpas_driver
     
   or for users with less IT knowledge better to use Jupyter notebook. You can try 
   by clicking the binder badge. This will open the project in JupyterLab in an 
   internet explorer and you can find all the examples on how to use co2mpas_driver. 
   The detailed explanation can be found in each page.
   
       
   **Contributing**
   
   Pull requests and stars are very welcome.
   
   For bugs and feature requests, please [create an issue](https://github.com/ashenafimenza/new_MFC/issues/new).
               
[1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
[2]: https://black.readthedocs.io/  