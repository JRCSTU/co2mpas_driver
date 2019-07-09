import os
import matplotlib.pyplot as plt
from new_MFC.common import reading_n_organizing as rno
from new_MFC.common import vehicle_functions as vf
from new_MFC.common import gear_functions as fg
from new_MFC.common import plot_templates as pt


def simple_run(db_name):
    # db_name = '../db/EuroSegmentCar'
    car_id = 27748
    gs_style = 1

    # file path without extension of the file
    db_name = os.path.dirname(db_name) + '/' + \
            os.path.splitext(os.path.basename(db_name))[0]

    db = rno.load_db_to_dictionary(db_name)

    selected_car = rno.get_vehicle_from_db(db, car_id)

    full_load_speeds, full_load_torque = vf.get_load_speed_n_torque(selected_car)
    speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(
        selected_car, full_load_speeds, full_load_torque)

    coefs_per_gear = vf.get_tan_coefs(speed_per_gear, acc_per_gear, 2)
    pt.plot_speed_acceleration_from_coefs(coefs_per_gear, speed_per_gear,
                                          acc_per_gear)

    cs_acc_per_gear = vf.get_cubic_splines_of_speed_acceleration_relationship(
        selected_car, speed_per_gear, acc_per_gear)
    Start, Stop = vf.get_start_stop(selected_car, speed_per_gear, acc_per_gear,
                                    cs_acc_per_gear)

    Tans = fg.find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop)

    gs = fg.gear_points_from_tan(Tans, gs_style, Start, Stop)

    for gear in gs:
        plt.plot([gear, gear], [0, 5], 'k')

    plt.show()


if __name__ == '__main__':
    simple_run()