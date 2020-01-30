from co2mpas_driver import dsp as driver
import matplotlib.pyplot as plt
import numpy as np


def simple_run():
    """
        parameters of the simulation
    :return:
    """
    veh_ids = [39393, 8188, 40516, 35452, 40225, 7897, 7972, 41388, 5766,
               9645, 9639, 5798, 8280, 34271, 34265, 6378, 39723, 34092, 2592,
               5635, 5630, 7661, 7683]
    v_des = 124/3.6
    v_start = 0
    dt = 0.1
    times = np.arange(0, 100, dt)

    vehicles = [driver(dict(vehicle_id=i,
                         inputs=dict(inputs={'gear_shifting_style': 1,
                                             'driver_style': 1,
                                             'starting_velocity': 0,
                                             'duration': 100, 'sim_start': 0,
                                             'sim_step': dt})))
                ['outputs']['driver_simulation_model'] for i in veh_ids]
    res = {}
    for myt in times:
        for my_veh in vehicles:
            if myt == times[0]:
                my_veh.reset(v_start)
                res[my_veh] = {'accel': [0], 'speed': [v_start], 'position': [0], 'gear': [0]}
                continue

            gear, next_velocity, acc, position = my_veh(dt, v_des)
            res[my_veh]['accel'].append(acc)
            res[my_veh]['speed'].append(next_velocity)
            res[my_veh]['gear'].append(gear)
            res[my_veh]['position'].append(position)

    for my_veh in vehicles:
        plt.figure('Acceleration-Speed')
        plt.plot(res[my_veh]['speed'], res[my_veh]['accel'])
        plt.show()
    return 0


simple_run()