from co2mpas_driver import dsp as driver
import matplotlib.pyplot as plt
import numpy as np


def simple_run():
    """
    Test vehicle simulation.

    :return:
    """
    veh_ids = [39393, 8188, 40516, 35452, 40225, 7897, 7972, 41388, 5766,
               9645, 9639, 5798, 8280, 34271, 34265, 6378, 39723, 34092, 2592,
               5635, 5630, 7661, 7683]
    v_des = 124/3.6
    v_start = 0
    # coefficients of resistances in case provided by user
    f0 = 200
    f1 = 0.2
    f2 = 0.005
    dt = 0.1
    times = np.arange(0, 100, dt)

    vehicles = [(i, driver(dict(vehicle_id=i, inputs=dict(inputs=dict(
        f0=f0, f1=f1, f2=f2,
        gear_shifting_style=0.8, driver_style=0.8, starting_velocity=0,
        duration=100, sim_start=0, sim_step=dt, use_linear_gs=True,
        use_cubic=False))))['outputs']['driver_simulation_model']) for i in
                veh_ids]
    res = {}
    for myt in times:
        for veh_id, my_veh in vehicles:
            if myt == times[0]:
                my_veh.reset(v_start)
                res[my_veh] = {'accel': [0], 'speed': [v_start], 'gear': [0]}
                continue

            gear, gear_count, next_velocity, acc = my_veh(dt, v_des)
            res[my_veh]['accel'].append(acc)
            res[my_veh]['speed'].append(next_velocity)
            res[my_veh]['gear'].append(gear)

    for car_id, my_veh in vehicles:
        plt.figure('Acceleration-Speed')
        plt.plot(res[my_veh]['speed'], res[my_veh]['accel'])
        plt.show()
    return 0


if __name__ == '__main__':
    simple_run()