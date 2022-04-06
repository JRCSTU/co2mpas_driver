# -*- coding: utf-8 -*-
#
# Copyright 2019 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl


class Driver:
    """
        Blueprint for driver.
    """

    def __init__(self, vehicle_mass, car_type, final_drive, gearbox_type, max_power,
                 fuel_type, type_of_car, car_width, car_height, transmission, gs, curves,
                 curves_dec, driver_style, r_dynamic=None, gear_box_ratios=None,
                 engine_max_torque=None, fuel_eng_capacity=None, fuel_engine_stroke=None,
                 fuel_turbo=None):
        self.vehicle_mass = vehicle_mass
        self.r_dynamic = r_dynamic
        self.car_type = car_type
        self.final_drive = final_drive
        self.gear_box_ratios = gear_box_ratios
        self.gearbox_type = gearbox_type
        self.engine_max_torque = engine_max_torque
        self.fuel_eng_capacity = fuel_eng_capacity
        self.max_power = max_power
        self.fuel_engine_stroke = fuel_engine_stroke
        self.fuel_type = fuel_type
        self.type_of_car = type_of_car
        self.car_width = car_width
        self.car_height = car_height
        self.fuel_turbo = fuel_turbo

        self.transmission = transmission
        self.gs = gs
        self.curves = list(curves)
        self.curves_dec = list(curves_dec)
        self.driver_style = driver_style
        self._velocity = self.position = self._gear_count = self._gear = None

    def reset(self, starting_velocity):
        from .simulation import gear_for_speed_profiles as func
        self._gear, self._gear_count = func(self.gs, starting_velocity, 0, 0)
        self._gear_count, self.position, self._velocity = 0, 0, starting_velocity
        return self

    def update(self, next_velocity, gear, gear_cnt, sim_step):
        from .simulation import gear_for_speed_profiles as func
        g, gc = func(self.gs, next_velocity, gear, gear_cnt)
        cnt = int(divmod(10 * 0.5, 10 * sim_step)[0])
        is_between = 0 < gear_cnt < cnt
        if g == gear and is_between:
            gc += 1
        elif g == gear and gear_cnt in [0, cnt]:
            gc = gear_cnt
        self._gear, self._gear_count, self._velocity = g, gc, next_velocity
        return g, gc

    def __call__(self, dt, desired_velocity, update=True):
        from .simulation import (
            gear_for_speed_profiles, accMFC, correct_acc_clutch_on
        )
        g, gc = gear_for_speed_profiles(
            self.gs, self._velocity, self._gear, self._gear_count
        )

        a = correct_acc_clutch_on(gc, accMFC(
            self._velocity, self.driver_style, desired_velocity,
            self.curves[g - 1],
        ), self.transmission)
        v = self._velocity + a * dt
        if update:
            self._gear, self._gear_count, self._velocity = g, gc, v
        return g, gc, v, a

    def redefine_ds(self, dt, desired_velocity, ids_new, update=True):
        from .simulation import (
            gear_for_speed_profiles, accMFC, correct_acc_clutch_on
        )
        g, gc = gear_for_speed_profiles(
            self.gs, self._velocity, self._gear, self._gear_count
        )

        a = correct_acc_clutch_on(gc, accMFC(
            self._velocity, ids_new, desired_velocity,
            self.curves[g - 1],
        ), self.transmission)
        v = self._velocity + a * dt
        s = self.position + self._velocity * dt + 0.5 * a * dt ** 2
        if update:
            self._gear, self._gear_count, self._velocity, self.position = g, gc, v, s
        return g, s, v, a

    def calculate_fuel_consumption(self, speed, acceleration, gear, gear_count, sim_step, slope=0):
        import math
        from co2mpas_driver.common import functions as func, vehicle_specs_class as vcc
        from .co2mpas import estimate_f_coefficients

        if self.fuel_type == 'electricity':
            return 0

        road_loads = estimate_f_coefficients(self.vehicle_mass, self.type_of_car, self.car_width,
                                             self.car_height)

        hardcoded_params = vcc.hardcoded_params()
        if self.gearbox_type == 'manual':
            veh_params = hardcoded_params.params_gearbox_losses['Manual']
            gb_type = 0
        else:
            veh_params = hardcoded_params.params_gearbox_losses['Automatic']
            gb_type = 1

        n_wheel_drive = self.car_type

        # The power on wheels in kW
        veh_wheel_power = func.calculate_wheel_power(speed, acceleration, road_loads, self.vehicle_mass, slope)

        # The speed on the wheels in [RPM]
        veh_wheel_speed = func.calculate_wheel_speeds(speed, self.r_dynamic)

        # # The torque on the wheels in [N*m]
        veh_wheel_torque = func.calculate_wheel_torques(veh_wheel_power, veh_wheel_speed)

        # Calculates final drive speed in RPM
        final_drive_speed = func.calculate_final_drive_speeds_in(veh_wheel_speed, self.final_drive)

        # Final drive torque losses [N*m].
        final_drive_torque_losses = func.calculate_final_drive_torque_losses_v1(n_wheel_drive, veh_wheel_torque,
                                                                                self.final_drive,
                                                                                hardcoded_params.final_drive_efficiency)

        # Final drive torque in [N*m].
        final_drive_torque_in = func.calculate_final_drive_torques_in(veh_wheel_torque, self.final_drive,
                                                                      final_drive_torque_losses)

        gear_box_speeds_in = func.calculate_gear_box_speeds_in_v1(gear, final_drive_speed, self.gear_box_ratios, 0)

        gearbox_params = func.create_gearbox_params(veh_params, self.engine_max_torque)

        gear_box_torques_in = func.gear_box_torques_in(hardcoded_params.min_engine_on_speed, final_drive_torque_in,
                                                       gear_box_speeds_in, final_drive_speed, gearbox_params,
                                                       gear_count)

        gear_box_power_out = 2 * math.pi * gear_box_torques_in * gear_box_speeds_in / 60000

        # gear_box_power_out = func.calculate_gear_box_power_out(gear_box_torques_in,
        #                                                        gear_box_speeds_in)

        br_eff_pres = func.calculate_brake_mean_effective_pressures(gear_box_speeds_in, gear_box_power_out,
                                                                    self.fuel_eng_capacity,
                                                                    hardcoded_params.min_engine_on_speed)

        engine_cm = func.mean_piston_speed(gear_box_speeds_in, self.fuel_engine_stroke)

        params = func.parameters(self.max_power, self.fuel_eng_capacity, self.fuel_type, self.fuel_turbo)
        fuel_A, fuel_B, fuel_C = func.calculate_fuel_ABC(params, engine_cm, br_eff_pres, 100)

        if br_eff_pres > 20:
            # Control for unrealistic Break Mean Effective Pressure values.
            print('BMEP> %.2f bar, EngineCM: %.2f, Gear: %d : Check out the MFC output. The engine will blow up!!!!' % (
                br_eff_pres, engine_cm, gear))

        if br_eff_pres > -0.5:
            # Fuel mean effective pressure
            VMEP = func.calculate_VMEP(fuel_A, fuel_B, fuel_C)
        else:
            VMEP = 0
        lower_heating_value = hardcoded_params.LHV[self.fuel_type]

        # Fuel consumption in grams.
        fc = func.calc_fuel_consumption(VMEP, self.fuel_eng_capacity,
                                        lower_heating_value, gear_box_speeds_in,
                                        sim_step)

        return fc
