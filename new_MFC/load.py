import logging
import os.path as osp
from new_MFC.common import reading_n_organizing as rno
import schedula as sh

log = logging.getLogger(__name__)
dsp = sh.Dispatcher(name='load')


def check_ext(fpath, *args, ext=('xls', 'xlsx')):
    return osp.splitext(fpath)[1][1:] in ext


@sh.add_function(dsp, outputs=['raw_data'], input_domain=check_ext)
def read_excel(input_path):
    """
    Read excel file.

    :param input_path:
        Excel input file path.
    :type input_path: str

    :return:
        Raw data.
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


@sh.add_function(dsp, outputs=['vehicle_id'])
def get_vehicle_id(raw_data):
    """
    Get vehicle ID from raw data.

    :param raw_data:
        Raw data.
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
    Get data base file path.

    :param raw_data:
        Raw data.
    :type raw_data: dict

    :return:
        Data base file path.
    :rtype: str
    """
    return raw_data.get('config', {}).get('db_path', sh.NONE)


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
        Raw data.
    :type raw_data: dict

    :param inputs:
        Inputs
    :type inputs:

    :return:
        Inputs
    :rtype: dict
    """
    d = {'vehicle_inputs': vehicle_inputs}
    return sh.combine_nested_dicts(d, raw_data, inputs, depth=2)


if __name__ == '__main__':
    # db_path = osp.join(osp.dirname(__file__), 'db', 'EuroSegmentCar.csv')
    input_path = 'C:/Apps/new_MFC/new_MFC/template/sample.xlsx'
    inputs = {
        'inputs': {'gear_shifting_style': 0.7},
        'vehicle_inputs': {'vehicle_mass': 0.4},
        'time_series': {'times': list(range(2, 23))}
    }
    raw_data = read_excel(input_path)
    vehicle_id = get_vehicle_id(raw_data)
    db_path = osp.dirname(__file__) + get_db_path(raw_data)
    vehicle_db = load_vehicle_db(db_path)
    vehicle_inputs = get_vehicle_inputs(vehicle_id, vehicle_db)
    data = merge_data(vehicle_inputs, raw_data, inputs)
    # sh.map_dict({})
    dsp.plot()
