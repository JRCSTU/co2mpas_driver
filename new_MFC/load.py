import copy
import logging
import numpy as np
import os.path as osp
import schedula as sh

log = logging.getLogger(__name__)
dsp = sh.BlueDispatcher(name='load')


def check_ext(fpath, *args, ext=('xls', 'xlsx')):
    return osp.splitext(fpath)[1][1:] in ext


@sh.add_function(dsp, outputs=['raw_data'], input_domain=check_ext)
def read_excel(input_path):
    """
    Read input file.

    :param input_path:
        Input file path.
    :type input_path: str

    :return:
        Raw data of input file.
    :rtype: dict
    """
    import xlrd
    import pandas as pd
    raw_data = {}
    with pd.ExcelFile(input_path) as xl:
        for k in ('inputs', 'config', 'vehicle_inputs', 'time_series'):
            try:
                if k == 'time_series':
                    raw_data[k] = xl.parse(k).dropna(1).to_dict(orient='list')
                else:
                    df = xl.parse(k, header=None, index_col=0, usecols=[0, 1]).T
                    raw_data[k] = df.dropna(1).to_dict(orient='records')[0]
            except IndexError:
                log.warning('Sheet `%s` is not well formatted!' % k)
            except xlrd.biffh.XLRDError:
                log.warning('Missing sheet (`%s`)!' % k)

    return raw_data

dsp.add_data('raw_data', {}, sh.inf(1, 0))

@sh.add_function(dsp, outputs=['vehicle_id'])
def get_vehicle_id(raw_data):
    """
    Get vehicle ID from raw data.

    :param raw_data:
        Raw data of input file.
    :type raw_data: dict

    :return:
        Vehicle ID.
    :rtype: int
    """
    return raw_data['config']['vehicle_id']


dsp.add_data(
    'db_path', osp.join(osp.dirname(__file__), 'db', 'EuroSegmentCar.csv'),
    sh.inf(1, 0)
)


@sh.add_function(dsp, outputs=['db_path'])
def get_db_path(raw_data):
    """
    Get data base file path from raw data.

    :param raw_data:
        Raw data of input file.
    :type raw_data: dict

    :return:
        Data base file path.
    :rtype: str
    """
    return raw_data.get('config', {}).get('db_path', sh.NONE)


_db_map = {
    "Transmission  / Gear ratio-Final drive": 'final_drive',
    "Transmission  / Gear ratio-Gear Box Ratios": "gear_box_ratios",
    'Weights-Empty mass': 'vehicle_mass',
    'Performance-Top speed': 'vehicle_max_speed',
    'General Specifications-Carbody': 'type_of_car',
    'Exterior sizes-Width': 'car_width',
    'Exterior sizes-Height': 'car_height',
    'Weights-Unladen mass': 'kerb_weight',
    'Exterior sizes-Wheelbase': 'wheelbase',
    'Drive-Wheel drive': 'car_type', 'Drive-Fuel': 'fuel_type',
    'Chassis-Rolling Radius Dynamic': 'r_dynamic',
    'Fuel Engine-Max torque': 'engine_max_torque',
    'Fuel Engine-Stroke': 'fuel_engine_stroke',
    'Drive-Total max power': 'max_power',
    'Fuel Engine-Turbo': 'fuel_turbo',
    'Fuel Engine-Capacity': 'fuel_eng_capacity',
    'General Specifications-Transmission': 'gearbox_type',
    'Fuel Engine-Max power': 'engine_max_power',
    'Electric Engine-Total max power': 'motor_max_power',
    'Electric Engine-Max torque': 'motor_max_torque',
    'Chassis-Rolling Radius Static': 'tyre_radius',
    "Fuel Engine-Max power RPM": "engine_max_speed_at_max_power"
}


@sh.add_function(dsp, outputs=['vehicle_db'])
def load_vehicle_db(db_path):
    """
    Load vehicle data base.

    :param db_path:
        Data base file path.
    :type db_path: str

    :return:
        Vehicle database.
    :rtype: dict
    """
    import pandas as pd
    df = pd.read_csv(db_path, encoding="ISO-8859-1", index_col=0)
    df = df[list(_db_map)].rename(columns=_db_map)

    df['gear_box_ratios'] = df['gear_box_ratios'].fillna('[]').apply(
        lambda x: [float(v) for v in x[1:-1].split('-') if v != '']
    )
    df.loc[df['fuel_type'] == 'petrol', 'ignition_type'] = 'positive'
    df.loc[df['fuel_type'] == 'diesel', 'ignition_type'] = 'compression'
    b = df['fuel_type'] == 'electricity'
    df.loc[b, ['ignition_type', 'gear_box_ratios']] = np.nan
    df['tyre_radius'] /= 1000  # meters.
    df['driveline_slippage'] = 0

    b = df['gearbox_type'] == 'automatic'
    b |= df['gearbox_type'] == 'single-speed fixed gear'
    df['transmission'] = np.where(b, 'automatic', 'manual')
    df['driveline_efficiency'] = np.where(b, .9, .93)

    df['vehicle_max_speed'] = (df['vehicle_max_speed'] / 3.6).values.astype(int)
    df['type_of_car'] = df["type_of_car"].str.strip()
    r = np.where(df['car_type'] == 'front', 2, 6)
    r[df['car_type'] == 'rear'] = 4
    df['car_type'] = r

    b = df['ignition_type'] == 'positive'
    df['idle_engine_speed_median'] = np.where(b, 750, 850)
    df['idle_engine_speed_std'] = 50
    df['r_dynamic'] /= 1000

    return df.to_dict('index')


@sh.add_function(dsp, outputs=['vehicle_inputs'])
def get_vehicle_inputs(vehicle_id, vehicle_db):
    """
    Get vehicle data.

    :param vehicle_id:
        Vehicle ID.
    :type vehicle_id: int

    :param vehicle_db:
        Vehicle database.
    :type vehicle_db: dict

    :return:
        Vehicle inputs.
    :rtype: dict
    """
    return vehicle_db[vehicle_id]


@sh.add_function(dsp, outputs=['data'])
def merge_data(vehicle_inputs, raw_data, inputs):
    """
    Merge data.

    :param vehicle_inputs:
        Vehicle inputs.
    :type vehicle_inputs: dict

    :param raw_data:
        Raw data of input file.
    :type raw_data: dict

    :param inputs:
        User inputs.
    :type inputs:

    :return:
        Merged data.
    :rtype: dict
    """
    d = {'vehicle_inputs': vehicle_inputs}
    d = sh.combine_nested_dicts(d, raw_data, inputs, depth=2)
    return sh.combine_dicts(
        d.pop('time_series', {}), d.pop('vehicle_inputs', {}), d['inputs']
    )


def format_data(data):
    """
    Format data.

    :param data:
        Data to be formatted.
    :type data: dict

    :return:
        Formatted data.
    :rtype: dict
    """
    data = copy.deepcopy(data)
    if isinstance(data.get('gear_box_ratios'), str):
        import json
        data['gear_box_ratios'] = json.loads(data['gear_box_ratios'])
    for k, v in list(data.items()):
        if isinstance(v, str):
            if v:
                continue
        elif not np.atleast_1d(np.isnan(v)).any():
            continue
        data.pop(k)
    return data


dsp.add_data('data', filters=[format_data])

if __name__ == '__main__':
    # raw_data = read_excel(input_path)
    # vehicle_id = get_vehicle_id(raw_data)
    # db_path = osp.dirname(__file__) + get_db_path(raw_data)
    # vehicle_db = load_vehicle_db(db_path)
    # vehicle_inputs = get_vehicle_inputs(vehicle_id, vehicle_db)
    #
    # data = merge_data(vehicle_inputs, raw_data, inputs)

    # db_path = osp.join(osp.dirname(__file__), 'db', 'EuroSegmentCar.csv')
    input_path = 'C:/Apps/new_MFC/new_MFC/template/sample.xlsx'
    inputs = {
        'inputs': {'gear_shifting_style': 0.7},
        'vehicle_inputs': {'vehicle_mass': 0.4},
        'time_series': {'times': list(range(2, 23))}
    }
    dsp(dict(input_path=input_path, inputs=inputs)).plot()
