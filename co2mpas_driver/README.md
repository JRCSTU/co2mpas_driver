## User guidelines for co2mpas_driver
<!--move them to CONTRIBUTING.md -->

This page contains user guidelines for first time users of co2mpas_driver 
library. It contains the explanations and definitions required to understand how to use
the library. These guidelines are written for users with less IT knowledge.
for more details on the new_MFC model https://journals.sagepub.com/doi/10.1177/0361198119838515

## Design diagram (core model)

   ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/core.png)

1. **Load module.** This model loads vehicle specifications based on the vehicle id
 and user input(gear shifting style, driver style, desired velocity) parameters 
 for the execution of simulation model in order to extract the drivers acceleration
 behavior as approaching the desired speed
   ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/load.png)
    * **Inputs** :
    
        1. db_path: file path for vehicle database based on the Euro car segment
           classification
        2. input_path: file path to an excel file that contains user input parameters
        3. inputs: users provide some parameters directly from their sample script
        4. vehicle_id: Id for a specific vehicle
    
    * **output** :
        
        1. data: this returns a data-value tree which is used as an input for 
           running simulation model. 

2. **Simulation Model.** 

    * **Run simulation:** This part simulates vehicles resulting acceleration per gear, gear shifting points, 
     final acceleration potential based on input parameters: gear shifting style, driver style and vehicle_id
     over the desired speed range.

3. **Installing new_MFC package**
    This package can be installed easily on any machine that has pip 
    from python package index using a requirement specifier 
    
        pip install new_MFC 

4. **How to use co2mpas_driver library**:
    In this example we will use new_MFC driver model in order to extract the drivers 
    acceleration behavior as approaching the desired speed.
    
    a. **Setup** 
       
      * First, set up python, numpy, matplotlib.
      
        set up python environment: numpy for numerical routines, and matplotlib 
        for plotting
        
            import numpy as np
            import matplotlib.pyplot as plt
       
      * Import dispatcher(dsp) from co2mpas_driver that contains functions 
        and simulation model to process vehicle data and Import also schedula
        for selecting and executing functions. for more information on how to use 
        schedula https://pypi.org/project/schedula/
         
            from co2mpas_driver import dsp
            import schedula as sh
       
    b. **Load data**
    
      * Load vehicle data for a specific vehicle from vehicles database
       
            db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar.csv'))
            
      * Load user input parameters from an excel file
       
            input_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                      'co2mpas_driver', 'template',
                                      'sample.xlsx'))     
      
      * Sample time series
       
            sim_step = 0.1 #The simulation step in seconds
            duration = 100 #Duration of the simulation in seconds
            times = np.arange(0, duration + sim_step, sim_step)
            
      * Load user input parameters directly writing in your sample script
       
            inputs = {
            'vehicle_id': 35135,  # A sample car id from the database
            'inputs': {'gear_shifting_style': 0.7, 'starting_speed': 0,
                       'desired_velocity': 40,
                       'driver_style': 1},  # gear shifting can take value
            # from 0(timid driver) to 1(aggressive driver)
            'time_series': {'times': times}
            }
            
    c. **Dispatcher**      
      
      * Dispatcher will select and execute the proper functions for the given inputs 
        and the requested outputs
               
            core = dsp(dict(db_path=db_path, input_path=input_path, inputs=inputs),
               outputs=['outputs'], shrink=True)
               
      * Plot workflow of the core model from the dispatcher
               
            core.plot()
            
        ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/core_example.PNG)
        
        **The Load module**
        
        ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/load_example.PNG)
        
        **merged vehicle data for the vehicle_id used above**
        
        ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/data.PNG)
            
      * Load outputs of dispatcher
        Select the chosen dictionary key (outputs) from the given dictionary.
               
            outputs = sh.selector(['outputs'], sh.selector(['outputs'], core))
            
      * select the desired output
            
            output = sh.selector(['Curves', 'poly_spline', 'Start', 'Stop', 'gs',
                          'discrete_acceleration_curves', 'velocities',
                          'accelerations', 'transmission'], outputs['outputs'])
             
        The final acceleration curvers (Curves), the engine acceleration potential 
        curves (poly_spline), before the calculation of the resistances and the
        limitation due to max possible acceleration (friction).
                        
            curves, poly_spline, start, stop, gs, discrete_acceleration_curves, \
            velocities, accelerations, transmission, discrete_acceleration_curves = \
            output['Curves'], output['poly_spline'], output['Start'], output['Stop'], output['gs'], \
            output['discrete_acceleration_curves'], output['velocities'], \
            output['accelerations'], output['transmission'], \
            output['discrete_acceleration_curves']
               
    c. **Plot**          
            
            plt.figure('Time-Speed')
            plt.plot(times, velocities)
            plt.grid()
            plt.figure('Speed-Acceleration')
            plt.plot(velocities, accelerations)
            plt.grid()
            plt.figure('Acceleration-Time')
            plt.plot(times, accelerations)
            plt.grid()
            
            
            plt.figure('Speed-Acceleration')
            for curve in discrete_acceleration_curves:
                sp_bins = list(curve['x'])
                acceleration = list(curve['y'])
                plt.plot(sp_bins, acceleration, 'k')
            plt.show()
            
    d. **Results**
      
     ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/speed-time.PNG)
     
     **Figure 1.** Speed(m/s) versus time(s) graph over the desired speed range.
     
     Acceleration(m/s*2) versus speed(m/s) graph
      
     ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/acce-speed.PNG)
     
     **Figure 2.** Acceleration per gear, the gear-shifting points and final acceleration potential of our selected 
       vehicle over the desired speed
     
     Acceleration(m/s*2) versus speed graph(m/s)
      
     ![alt text](https://github.com/ashenafimenza/new_MFC/blob/master/co2mpas_driver/images/acc-time.PNG)
      
     **Figure 3.** The final acceleration potential of our selected vehicle over the desired speed range
     
            return 0
            
            if __name__ == '__main__':
                simple_run()  
               
[1]: https://ljvmiranda921.github.io/notebook/2018/06/21/precommits-using-black-and-flake8/
[2]: https://black.readthedocs.io/  