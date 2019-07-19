import schedula as sh

dsp = sh.Dispatcher()
dsp.add_function(
    function_id='load',
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
