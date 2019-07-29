import schedula as sh
from new_MFC.load import dsp as _load
from new_MFC.process import dsp as _process
dsp = sh.Dispatcher()
dsp.add_dispatcher(
    dsp=_load,
    inputs=['inputs', 'vehicle_id', 'db_path', 'input_path'],
    outputs=['data']
)
# dsp.add_function(
#     function=sh.SubDispatch(_process),
#     inputs=['data'],
#     outputs=['outputs']
# )
dsp.add_function(
    function=sh.SubDispatchFunction(_process, function_id='simulation', inputs=['type_of_car', 'car_type', 'veh_max_speed', 'veh_mass',
                                                      'engine_max_power', 'car_width', 'gear_box_ratios', 'car_height',
                                                      'use_linear_gs', 'speed_per_gear', 'gs_style', 'acc_per_gear',
                                                      'degree', 'use_cubic', 'use_cubic'],
                                    outputs=['Curves', 'poly_spline', 'Start', 'Stop', 'gs']),
    inputs=['data'],
    outputs=['outputs']
)
dsp.add_function(
    function_id='write',
    inputs=['output_path', 'outputs']
)
dsp.add_function(
    function_id='plot',
    inputs=['output_plot_folder', 'outputs']
)
if __name__ == '__main__':
    dsp.plot()
