## co2mpas_driver model
<!--move them to CONTRIBUTING.md -->

This page contains the explanations, definitions and examples required to understand
and use co2mpas_driver (i.e MFC) model.

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



   
   **Contributing**
   
   Pull requests and stars are very welcome.
   
   For bugs and feature requests, please [create an issue](https://github.com/ashenafimenza/new_MFC/issues/new).
               
[1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
[2]: https://black.readthedocs.io/  