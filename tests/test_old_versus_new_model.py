from co2mpas_driver.common import simulation_part as sp
import numpy as np
import matplotlib.pyplot as plt
from co2mpas_driver.common import curve_functions as mf
from co2mpas_driver.common import reading_n_organizing as rno
from co2mpas_driver.common import gear_functions as fg
from co2mpas_driver import dsp as driver
import random


def simple_run():
    ''':parameters of the simulation'''
    # Vehicle databased based on the Euro Car Segment classification
    db_name = '../co2mpas_driver/db/EuroSegmentCar_cleaned'

    veh_ids = [39393, 8188, 40516, 35452, 40225, 7897, 7972, 41388, 5766, 9645,
               9639, 5798, 8280, 34271, 34265, 6378, 39723, 34092, 2592, 5635,
               5630, 7661, 7683]
    v_des = [np.round(random.randint(45, 150) / 3.6, 2) for i in
             range(len(veh_ids))]
    ds = [np.round(random.random(), 1) for i in range(len(veh_ids))]
    gs_style = [np.round(random.random(), 1) for i in range(len(veh_ids))]
    v_start = [random.randint(0, 10) for i in range(len(veh_ids))]
    dt = 0.1
    duration = 100
    times = np.arange(0, duration, dt)
    # **********************************************************
    vehicles = [driver(dict(vehicle_id=i,
                            inputs=dict(inputs={'gear_shifting_style': gs_style[idx],
                                                'starting_velocity': v_start[idx],
                                                'driver_style': ds[idx],
                                                'desired_velocity': v_des[idx],
                                                'sim_start': 0,
                                                'sim_step': dt,
                                                'duration': duration
                                                })))
                ['outputs'] for idx, i in
                enumerate(veh_ids)]

    # **********************************************************
    '''import vehicle object, curves and gear shifting strategy'''
    db = rno.load_db_to_dictionary(db_name)

    for idx, i in enumerate(veh_ids):
        selected_car = rno.get_vehicle_from_db(db, i)
        Curves, cs_acc_per_gear, StartStop, gs = mf.gear_4degree_curves_with_linear_gs(selected_car, gs_style[idx])
        Speeds = [v_start[idx]]
        Accelerations = [0]
        speed = v_start[idx]
        gear, gear_count = fg.gear_for_speed_profiles(gs, speed, 0, 0)
        gear_count = 0

        '''Core loop'''
        for t in times:
            speed, gear, gear_count = sp.simulation_step_function(selected_car, speed, gear, gear_count, gs, Curves, v_des[idx], ds[idx], dt)
            '''Gather data'''
            Speeds.append(speed)
            Accelerations.append((Speeds[-1] - Speeds[-2])/dt)
        fig1 = plt.figure('Acceleration-Speed')
        plt.plot(Speeds, Accelerations)
        plt.show()

        fig1.savefig(
            f"testfigs/veh_{veh_ids[idx]}_acceleration_speed_old_model_des_{np.round(v_des[idx], 2)}_vstart_{np.round(v_start[idx], 2)}_ds_{ds[idx]}_gs_{gs_style[idx]}.png",
            dpi=150)

    for idx, my_veh in enumerate(vehicles):
        fig2 = plt.figure('Acceleration-Speed')
        plt.plot(my_veh['velocities'], my_veh['accelerations'])
        plt.show()

        fig2.savefig(
            f"testfigs/veh_{veh_ids[idx]}_acceleration_speed_new_model_des_{np.round(v_des[idx], 2)}_vstart_{np.round(v_start[idx], 2)}_ds_{ds[idx]}_gs_{gs_style[idx]}.png",
            dpi=150)
    return 0


simple_run()
