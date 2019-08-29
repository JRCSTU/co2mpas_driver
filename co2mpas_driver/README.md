## User guidelines for co2mpas_driver
<!--move them to CONTRIBUTING.md -->

This page contains user guidelines intended for first time users of co2mpas_driver 
library. It contains the explanations and definitions required to understand how to use
the the library. These guidelines are written for users without specific IT knowledge.
for more details https://journals.sagepub.com/doi/10.1177/0361198119838515

## Design diagram

   ![alt text](images/design.png)

1. **Load module.** This model loads vehicle specifications based on the vehicle id
 and user input(gear shifting style, driver style, desired velocity) parameters 
 for the execution of simulation model in order to extract the drivers acceleration
 behavior as approaching the desired speed

    * **Inputs** :
    
        1. db_path: file path for vehicle database based on the Euro car segment
           classification
        2. input_path: file path to an excel file that contains user input parameters
        3. inputs: users provide some parameters directly from console
        4. vehicle_id: Id for a specific vehicle
    
    * **output** :
        
        1. data: this returns a data-value tree which is used as an input for 
           running simulation model. 

   ![alt text](images/load.png)

2. **Simulation Model.** 

    * **Run simulation:** This part runs simulation of 

3. **How to download and Install co2mpas_driver library**
    This package can be installed easily on any machine that has pip
    
        pip install new_MFC 

[1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
[2]: https://black.readthedocs.io/  