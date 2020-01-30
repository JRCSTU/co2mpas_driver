from co2mpas_driver import dsp as driver
import matplotlib.pyplot as plt
import numpy as np
import random


def simple_run():
    """
        parameters of the simulation
    :return:
    """

    '''
    problem try 15109 with ds gs = 1
    '''

    # veh_ids = [35135]
    # 44387 - electric problem
    # 26539, 26569,34603, 34565, 6382,47844
    veh_ids = [35135, 39393, 27748, 15109, 8183, 8188, 26629, 8358, 17145, 17146, 40516, 35452, 40225, 7897,
               7972, 41388, 35361, 5768, 5766, 3408, 15552, 9645, 9639, 8620, 8592, 5779, 5798, 8280, 8267, 4396, 4416,
               34271, 34265, 6378, 39723, 34092, 2508, 2592, 5635, 5630, 34499,
               34474, 7661, 7683, 8709, 9769, 20409, 10133, 26765, 1872, 10328, 10349, 35476, 41989, 26799, 26851,
               27189, 27096, 23801, 3079, 36525, 47766, 6386, 6390, 18771, 18767, 2090, 1978, 33958, 33986, 5725, 5718,
               36591, 4350, 39396, 40595, 5909, 5897, 5928, 5915, 40130, 42363, 34760, 34766, 1840, 1835, 36101, 42886,
               1431, 24313, 46547, 44799, 41045, 39820, 3231, 3198, 34183, 34186, 20612, 20605, 1324, 9882, 9885, 4957,
               44856, 18195, 5595, 5603, 18831, 18833, 22376, 9575, 5391, 5380, 9936, 7995, 6331, 18173, 43058, 34286,
               34279, 20699, 20706, 34058, 34057, 24268, 24288, 19028, 19058, 7979, 7976, 22563, 22591, 34202, 34196,
               40170, 44599, 5358, 5338, 34024, 34015, 7836, 7839, 9738, 9754, 9872, 9856, 6446, 8866, 9001, 8996, 9551,
               6222]

    veh_ids = [39393, 8188, 40516, 35452, 40225, 7897, 7972, 41388, 5766, 9645,
               9639, 5798, 8280, 34271, 34265, 6378, 39723, 34092, 2592, 5635,
               5630, 7661, 7683]
    v_des = [np.round(random.randint(45, 150) / 3.6, 2) for i in range(len(veh_ids))]
    ds = [np.round(random.random(), 1) for i in range(len(veh_ids))]
    gs = [np.round(random.random(), 1) for i in range(len(veh_ids))]
    v_start = [random.randint(0, 10) for i in range(len(veh_ids))]
    dt = 0.1
    times1 = np.arange(0, 50, dt)
    times2 = np.arange(0, 50, dt)
    times3 = np.arange(0, 50, dt)

    # '''
    # test which vehicles work'''
    # problem_ids = []
    # ok_ids = []
    # complete = 0
    # cnt = 0
    # while complete == 0:
    #     try:
    #         # One by one the vehicle ids it works. When you put a list there is a problem.
    #         test_ids = veh_ids[cnt:cnt]
    #         vehicles = [driver(dict(vehicle_id=i,
    #                      inputs=dict(inputs={'gear_shifting_style': gs[cnt],
    #                                          'driver_style': ds[cnt]})))
    #          ['outputs']['driver_simulation_model'] for i in test_ids]
    #         ok_ids.append(veh_ids[cnt])
    #     except:
    #         problem_ids.append(veh_ids[cnt])
    #     cnt += 1
    #     if cnt == len(veh_ids):
    #         complete = 1
    #
    # # end test
    # print(ok_ids)
    # print(problem_ids)
    # exit()

    vehicles = [driver(dict(vehicle_id=i,
                            inputs=dict(inputs={'gear_shifting_style': gs[idx],
                                                'driver_style': ds[idx]})))
                ['outputs']['driver_simulation_model'] for idx, i in enumerate(veh_ids)]
    # exit()
    res = {}
    for myt in times1:
        for idx, my_veh in enumerate(vehicles):
            if myt == times1[0]:
                my_veh.reset(v_start[idx])
                res[my_veh] = {'accel': [0], 'speed': [v_start[idx]], 'position': [0], 'gear': [0]}
                continue

            gear, next_velocity, position, acc = my_veh(dt, v_des[idx])
            res[my_veh]['accel'].append(acc)
            res[my_veh]['speed'].append(next_velocity)
            res[my_veh]['gear'].append(gear)
            res[my_veh]['position'].append(position)
    for myt in times2:
        for my_veh in vehicles:
            gear, next_velocity, position, acc = my_veh(dt, v_des[idx] - 40 / 3.6)
            res[my_veh]['accel'].append(acc)
            res[my_veh]['speed'].append(next_velocity)
            res[my_veh]['gear'].append(gear)
            res[my_veh]['position'].append(position)
    for myt in times3:
        for my_veh in vehicles:
            gear, next_velocity, position, acc = my_veh(dt, v_des[idx] + 50 / 3.6)
            res[my_veh]['accel'].append(acc)
            res[my_veh]['speed'].append(next_velocity)
            res[my_veh]['gear'].append(gear)
            res[my_veh]['position'].append(position)

    for idx, my_veh in enumerate(vehicles):
        fig1 = plt.figure('Acceleration-Speed')
        plt.plot(res[my_veh]['speed'], res[my_veh]['accel'])
        plt.show()

        # fig1 = plt.figure('Speed-Time')
        # ax1 = fig1.add_subplot(111)
        # ax1.plot(res[my_veh]['speed'])
        # plt.show()
        fig1.savefig(
            f"../tests/testfigs/veh_{veh_ids[idx]}_acceleration_speed_des_{np.round(v_des[idx], 2)}_vstart_{np.round(v_start[idx], 2)}_ds_{ds[idx]}_gs_{gs[idx]}.png",
            dpi=150)

    return 0


simple_run()
