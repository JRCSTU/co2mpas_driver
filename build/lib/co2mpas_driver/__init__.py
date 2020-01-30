# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
Functions and a model `dsp` to processes a CO2MPAS input file.

Sub-Modules:
.. currentmodule:: co2mpas_driver
.. autosummary::
    :nosignatures:
    :toctree: _build/co2mpas_driver/
    load
    model
    write
    plot
"""
import matplotlib.pyplot as plt
import schedula as sh
from co2mpas_driver.load import dsp as _load
from co2mpas_driver.model import dsp as _model
from co2mpas_driver.plot import dsp as _plot

dsp = sh.Dispatcher()
dsp.add_dispatcher(
    dsp=_load,
    inputs=['inputs', 'vehicle_id', 'db_path', 'input_path'],
    outputs=['data']
)

dsp.add_function(
    function=sh.SubDispatch(_model),
    inputs=['data'],
    outputs=['outputs']
)

dsp.add_function(
    function_id='write',
    inputs=['output_path', 'outputs']
)

dsp.add_function(
    function=sh.SubDispatch(_plot),
    inputs=['output_plot_folder', 'outputs']
)

if __name__ == '__main__':
    sol = dsp(dict(vehicle_id=35135,
                           inputs=dict(inputs={'gear_shifting_style': 0.7,
                                               'starting_velocity': 0,
                                               'driver_style': 1,
                                               'desired_velocity': 124/3.6,
                                               'sim_start': 0, 'sim_step': 0.1,
                                               'duration': 100})))
    # velocity = sol['outputs']['velocities']
    # acceleration = sol['outputs']['accelerations']
    # plt.figure('Speed-Acceleration')
    # plt.plot(velocity[1:], acceleration[1:])
    # plt.grid()
    # plt.show()
    dsp.plot()
