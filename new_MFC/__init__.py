import schedula as sh
from new_MFC.load import dsp as _load
from new_MFC.process import dsp as _process
from new_MFC.plot import dsp as _plot

dsp = sh.Dispatcher()
dsp.add_dispatcher(
    dsp=_load,
    inputs=['inputs', 'vehicle_id', 'db_path', 'input_path'],
    outputs=['data']
)

dsp.add_function(
    function=sh.SubDispatch(_process),
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
    dsp.plot()
