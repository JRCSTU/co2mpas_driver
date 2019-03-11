import matplotlib.pyplot as plt
import reading_n_organizing as rno
import vehicle_functions as vf
import find_gear as fg


db_name = 'car_db_sample'
car_id = 24211
gs_style = 1

db = rno.load_db_to_dictionary(db_name)

selected_car = rno.get_vehicle_from_db(db, car_id)

full_load_speeds, full_load_torque = vf.get_load_speed_n_torque(selected_car)
speed_per_gear, acc_per_gear = vf.get_speeds_n_accelerations_per_gear(selected_car, full_load_speeds, full_load_torque)

coefs_per_gear = vf.get_tan_coefs(speed_per_gear,acc_per_gear,2)
vf.plot_speed_acceleration_from_coefs(coefs_per_gear,speed_per_gear, acc_per_gear )

cs_acc_per_gear = vf.get_cubic_splines_of_speed_acceleration_relationship(selected_car, speed_per_gear, acc_per_gear)
Start, Stop = vf.get_start_stop(selected_car, speed_per_gear, acc_per_gear, cs_acc_per_gear)

Tans = fg.find_list_of_tans_from_coefs(coefs_per_gear, Start, Stop)

gs = fg.gear_points_for_AIMSUN_tan(Tans,gs_style,Start,Stop)

for gear in gs:
    plt.plot([gear,gear],[0,5],'k')

plt.show()