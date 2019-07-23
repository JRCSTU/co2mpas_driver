import schedula as sh
from new_MFC.load import dsp as _load
dsp = sh.Dispatcher()
dsp.add_dispatcher(
    dsp=_load,
    inputs=['inputs', 'vehicle_id', 'db_path', 'input_path'],
    outputs=['data']
)
dsp.add_function(
    function_id='process',
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
