from os import chdir, path as osp
import matplotlib.pyplot as plt
from co2mpas_driver import dsp
import schedula as sh

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def simple_run():
    # Vehicles database path
    db_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                   'co2mpas_driver', 'db',
                                   'EuroSegmentCar.csv'))
    # input file path
    input_path = osp.abspath(osp.join(osp.dirname(my_dir + '/../'),
                                      'co2mpas_driver', 'template',
                                      'sample.xlsx'))
    # user inputs
    inputs = {
        'vehicle_id': 39393,  # A sample car id from the database
        'inputs': {'gear_shifting_style': 0.8},
        'vehicle_inputs': {'degree': 2},  # gear shifting can take value
        # from 0(timid driver) to 1(aggressive driver)
        'time_series': {'times': list(range(2, 23))}
    }

    core = dsp(dict(db_path=db_path, input_path=input_path, inputs=inputs),
               outputs=['outputs'], shrink=True)

    # plots simulation model
    core.plot()

    # outputs of the dispatcher
    outputs = sh.selector(['outputs'], sh.selector(['outputs'], core))

    # select the desired output
    output = sh.selector(['Curves', 'poly_spline', 'Start', 'Stop', 'gs',
                          'discrete_acceleration_curves'], outputs['outputs'])

    curves, poly_spline, start, stop, gs, discrete_acceleration_curves = \
        output['Curves'], output['poly_spline'], output['Start'], \
        output['Stop'], output['gs'], output['discrete_acceleration_curves']

    for d in discrete_acceleration_curves:
        plt.plot(d['x'], d['y'])

    plt.show()

    return 0


if __name__ == '__main__':
    simple_run()
