from co2mpas_driver import dsp


def simple_run():
    """
        parameters of the simulation
    :return:
    """
    veh_ids = [39393, 27748]
    v_des = 40
    v_start = 10
    import numpy as np
    times = np.arange(0, 10, 0.1)

    vehicles = [dsp(dict(vehicle_id=i,
                         inputs=dict(inputs={'gear_shifting_style': 0.9,
                                             'driver_style': 1})))
                ['outputs']['driver_simulation_model'] for i in veh_ids]

    for myt in times:
        print("******************")
        for my_veh in vehicles:
            vehicle_model = my_veh.reset(v_start)
            gear, next_velocity, acc = vehicle_model(myt, v_des)

            print(next_velocity)

    return 0


simple_run()