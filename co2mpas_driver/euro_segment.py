from os import path as osp, chdir
from tabulate import tabulate
from co2mpas_driver import load as ld

my_dir = osp.dirname(osp.abspath(__file__))
chdir(my_dir)


def print_vehicle_db():
    """
    Sample file printing an indicative set of columns from the dataset for all vehicles of a Euro-Segment.

    :return:
    """
    # Vehicle databased based on the Euro Car Segment classification
    db_path = osp.abspath(osp.join(my_dir, 'db', 'EuroSegmentCar.csv'))
    import pandas as pd
    df = pd.read_csv(db_path, encoding="ISO-8859-1", index_col=0)
    df = df[list(ld._db_map)].rename(columns=ld._db_map)
    df_ = df.loc[:, ld._db_map.values()]
    print(tabulate(df_, headers='keys', tablefmt='psql'))


if __name__ == '__main__':
    print_vehicle_db()