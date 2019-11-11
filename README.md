## Com2pas_driver: Try it live
<!--move them to CONTRIBUTING.md -->
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JRCSTU/co2mpas_driver/master?urlpath=lab/tree/examples)

Access this Binder at the following URL:

https://mybinder.org/v2/gh/ashenafimenza/binder_co2mpas_driver/master

Click the binder badge to try it live without installing anything. 
This will take you directly to the ipython notebook to present examples on 
how to use co2mpas_driver model (i.e., MFC) to simulate the driver behaviour of 
a vehicle.

## User guidelines for co2mpas_driver

This page contains user guidelines for first time users of co2mpas_driver 
library. It contains the explanations and definitions required to understand how to use
the library. These guidelines are written for users with less IT knowledge.
for more details on the new_MFC model https://journals.sagepub.com/doi/10.1177/0361198119838515

1. **Download and Install co2mpas_driver library**
    This package can be installed easily on any machine that has pip
    
        pip install co2mpas_driver
        
    or if you have access to the project under JRCSTU github repository then clone
    the co2mpas_driver project to your local machine.
   
        git clone https://github.com/JRCSTU/co2mpas_driver.git
       
2. **Install on your local machine**
    You can install on your machine using:
    
        pip install dist/co2mpas_driver-1.0.0-py2.py3-none-any.whl 
        
3. **In order to use co2mpas_driver library**
   you can start importing and using co2mpas driver on python console.
   
        import co2mpas_driver
     
   or for users with less IT knowledge better to use Jupyter notebook.  It is 
   easier to learn and use for first time users of co2mpas_driver library
   
   It is important to install jupyter notebook.
       
       pip install jupyter
       
   And then from your terminal change to your project directory and launch jupyter.
   
       jupyter notebook
       
   This will open the project in an internet explorer and you can find all the 
   examples in the Jupyter home page. The detailed explanation can be found in each page.
   
## Design diagram (core model)  
   
   ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/core.png)
   
1. **Load module.** This model loads vehicle data based on the vehicle id
 and user input(gear shifting style, driver style, desired velocity) parameters 
 for the execution of simulation model in order to extract the drivers acceleration
 behavior as approaching the desired speed
   ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/load.png)
    * **Inputs** :
    
        1. db_path: file path for vehicle database based on the Euro car segment
           classification.
        2. input_path: file path to an excel file that contains user input 
           parameters where the user can change parameters like driver style, 
           gear shifting style, time series, starting speed, desired velocity, 
           file path to the vehicle database etc.
        3. inputs: users also can provide input parameters directly from their 
           sample script in addition or instead of the sample excel file.
        4. vehicle_id: Id for a specific vehicle.
    
    * **output** :
        
        1. data: this returns a data-value tree which is used as an input for 
           executing different functions in the dispatcher for simulation model. 

2. **Simulation Model.** 
    
    * **Run simulation:** This part simulates vehicles resulting acceleration per gear, 
       gear shifting points, final acceleration potential based on input parameters: 
       gear shifting style, driver style and vehicle_id over the desired speed range.

4. **How to use co2mpas_driver library**:
    Yout can start importing co2mpas_driver.
         
            import co2mpas_driver
       
   **Contributing**
   
   Pull requests and stars are very welcome.
   
   For bugs and feature requests, please [create an issue](https://github.com/ashenafimenza/new_MFC/issues/new).
               
[1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
[2]: https://black.readthedocs.io/  