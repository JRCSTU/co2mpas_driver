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
    function=sh.SubDispatchFunction(_process, inputs=['engine_max_power',
                                                      'ignition_type',
                                                      'engine_max_speed_at_max_power',
                                                      'idle_engine_speed'],
                                    outputs=['full_load_powers', 'full_load_speeds']),
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
