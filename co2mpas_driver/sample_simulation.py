from os import path as osp, chdir
from co2mpas_driver import dsp as driver
import click
import matplotlib.pyplot as plt

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


@click.command()
@click.option('--db', type=str, default='db/EuroSegmentCar_cleaned.csv',
              help='vehicle database file path')
@click.option('--car_id', type=int, default=35135, help='vehicle id')
@click.option('--ds', type=float, default=1, help='driver style')
@click.option('--gs', type=float, default=0.7, help='gear shifting style')
@click.option('--v_des', type=float, default=124/3.6, help='desired velocity')
@click.option('--v_start', type=float, default=0, help='starting velocity')
@click.option('--sim_start', type=float, default=0, help='simulation step in seconds')
@click.option('--sim_step', type=float, default=0.1, help='simulation step in seconds')
@click.option('--duration', type=float, default=100, help='Duration of the simulation in seconds')
@click.option('--plot', type=bool, default=False, help='plots driver model work flow')
def run_simulation(db, car_id, ds, v_start, gs, v_des, sim_start, sim_step,
                   duration, plot):
    """
        This script runs simulation and plots the final acceleration versus
        velocity graph of the vehicle.
    :return:
    """
    driver_model = driver(dict(vehicle_id=car_id, db_path=db,
                           inputs=dict(inputs={'gear_shifting_style': gs,
                                               'starting_velocity': v_start,
                                               'driver_style': ds,
                                               'desired_velocity': v_des,
                                               'sim_start': sim_start, 'sim_step': sim_step,
                                               'duration': duration})))
    if plot:
        driver_model.plot()
    outputs = driver_model['outputs']
    discrete_acceleration_curves = outputs['discrete_acceleration_curves']
    fig = plt.figure()
    for curve in discrete_acceleration_curves:
        sp_bins = list(curve['x'])
        acceleration = list(curve['y'])
        plt.plot(sp_bins, acceleration)
    plt.plot(outputs['velocities'][1:], outputs['accelerations'][1:])
    plt.xlabel('Speed', fontsize=18)
    plt.ylabel('Acceleration', fontsize=16)
    plt.legend(['acceleration per gear 0', 'acceleration per gear 1',
                'acceleration per gear 2', 'acceleration per gear 3',
                'acceleration per gear 4', 'final acceleration'])
    plt.grid()
    plt.show()
    return 0


if __name__ == '__main__':
    run_simulation()
