import simulation_part as sp
import numpy as np
import matplotlib.pyplot as plt
import main_functions as mf
import reading_n_organizing as rno
import find_gear as fg


def simple_run():
    ''':parameters of the simulation'''
    db_name = 'db/delete_car_db_ICEV_EV'
    car_id = 26573
    gs_style = 0.2
    vdes = 350
    v_start = 0
    sim_step = 0.1
    driver_style = 1
    duration = 100  # sec
    times = np.arange(0, duration + sim_step, sim_step)

    '''import vehicle object, curves and gear shifting strategy'''
    db = rno.load_db_to_dictionary(db_name)
    selected_car = rno.get_vehicle_from_db(db, car_id)
    Curves, cs_acc_per_gear, StartStop, gs = mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style)
    # Curves, cs_acc_per_gear, StartStop, gs = mf.gear_curves_n_gs_from_poly(selected_car, gs_style,4)

    '''Lists to gather simulation data'''
    Speeds = [v_start]
    Acceleration = [0]

    '''Initialize speed and gear'''
    speed = v_start
    gear, gear_count = fg.gear_for_speed_profiles(gs, speed, 0, 0, selected_car.transmission)
    gear_count = 0

    '''Core loop'''
    for t in times:
        speed, gear, gear_count = sp.simulation_step_function(selected_car,speed,gear,gear_count,gs,Curves,vdes,driver_style,sim_step)

        '''Gather data'''
        Speeds.append(speed)
        Acceleration.append((Speeds[-1] - Speeds[-2])/sim_step)

    '''Plot'''
    plt.figure('Time-Speed')
    plt.plot(times, Speeds[1:])
    plt.grid()
    plt.figure('Speed-Acceleration')
    plt.plot(Speeds[1:], Acceleration[1:])
    plt.grid()
    plt.figure('Acceleration-Time')
    plt.plot(times,Acceleration[1:])
    plt.grid()

    plt.figure('Speed-Acceleration')
    for i,gear_curve in enumerate(Curves):
        sp_bins = np.arange(StartStop[0][i],StartStop[1][i]+0.1,0.1)
        accelerations = gear_curve(sp_bins)
        plt.plot(sp_bins,accelerations,'k')
    plt.grid()
    plt.show()
    return 0


simple_run()
