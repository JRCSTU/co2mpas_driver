from os import path as osp, chdir
import matplotlib.pyplot as plt
from co2mpas_driver import dsp as driver

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    """
    This example plots the acceleration and deceleration potential curves (over speed) of the vehicle for each gear.

    :return:
    """
    # A sample car id from the database = {
    car_id_db = {
        'fuel_engine': [35135, 39393, 27748, 8188, 40516, 35452, 40225, 7897, 7972, 41388, 5766, 9645, 9639, 5798, 8280,
                        34271, 34265, 6378, 39723, 34092, 2592, 5635, 5630, 7661, 7683, 8709, 9769, 1872, 10328, 35476,
                        41989, 26799, 26851, 27189, 23801, 3079, 36525, 47766, 6386, 33958, 33986, 5725, 5718, 36591,
                        4350, 39396, 40595, 5909, 5897, 5928, 5915, 40130, 42363, 34760, 34766, 1835, 36101, 42886,
                        1431, 46547, 44799, 41045, 39820, 34183, 34186, 20612, 20605, 1324, 9882, 4957, 5595, 18831,
                        18833, 9575, 5380, 9936, 7995, 6331, 18173, 34286, 34279, 20706, 34058, 34057, 24268, 19028,
                        19058, 7979, 22591, 34202, 40170, 44599, 5358, 5338, 34015, 9872, 9856, 6446, 8866, 9001, 9551,
                        6222],
        'electric_engine': [47844]
    }
    car_id = 35135

    # Core model, this will select and execute the proper functions for the given inputs and returns the output
    # You can also pass your vehicle's database path db_path='path to vehicle db'
    if car_id in car_id_db['electric_engine']:
        sol = driver(dict(vehicle_id=47844))['outputs']
    else:
        # The gear shifting style as described in the TRR paper.
        gs_style = 0.8
        # 4th degree polynomial
        degree = 4
        sol = driver(dict(vehicle_id=car_id, inputs=dict(inputs=dict(
            gear_shifting_style=gs_style, degree=degree, use_estimated_res=True,
            use_linear_gs=True, use_cubic=False))))['outputs']
    # Plots workflow of the core model, this will automatically open an internet browser and shows the work flow
    # of the core model. you can click all the rectangular boxes to see in detail sub models like load, model,
    # write and plot.
    # you can uncomment to plot the work flow of the model
    # driver.plot(1)  # Note: run your IDE as Administrator if file permission error.

    fig = plt.figure()
    for gear, curve in enumerate(sol['discrete_acceleration_curves'], start=1):
        sp_bins = list(curve['x'])
        acceleration = list(curve['y'])
        plt.plot(sp_bins, acceleration, label=f'gear {gear}')
    plt.legend()
    plt.text(27, 2.5, f'Simulation car:{car_id}')
    fig.suptitle('Acceleration over speed', x=0.54, y=1, fontsize=12)
    plt.xlabel('Speed (m/s)', fontsize=12)
    plt.ylabel('Acceleration (m/s2)', fontsize=12)
    plt.grid()
    #fig.savefig(f'sample_4degree_with_linear_Acceleration_Speed_for_vehicle_id_{car_id}_gear_shifting_style_{gs_style}.jpg')

    fig1 = plt.figure('Deceleration over Speed')
    for gear, curve in enumerate(sol['discrete_deceleration_curves'], start=1):
        sp_bins = list(curve['x'])
        deceleration = list(curve['y'])
        plt.plot(sp_bins, deceleration, label=f'gear {gear}')
    plt.legend()
    plt.text(23, 2.5, f'Simulation car:{car_id}')
    fig1.suptitle('Deceleration over Speed', x=0.54, y=1, fontsize=12)
    plt.xlabel('Speed (m/s)', fontsize=12)
    plt.ylabel('Deceleration (m/s2)', fontsize=12)
    plt.grid()
    #fig1.savefig(f'sample_4degree_with_linear_Deceleration_Speed_for_vehicle_id_{car_id}_gear_shifting_style_{gs_style}.jpg')

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
