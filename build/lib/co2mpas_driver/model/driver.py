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

    def __init__(self, transmission, gs, curves, driver_style):
        self.transmission = transmission
        self.gs = gs
        self.curves = list(curves)
        self.driver_style = driver_style
        self._velocity = self.position = self._gear_count = self._gear = None

    def reset(self, starting_velocity):
        from .simulation import gear_for_speed_profiles as func
        self._gear = func(self.gs, starting_velocity, 0, 0)[0]
        self._gear_count, self.position, self._velocity = 0, 0, starting_velocity
        return self

    def __call__(self, dt, desired_velocity, update=True):
        from .simulation import (
            gear_for_speed_profiles, accMFC, correct_acc_clutch_on
        )
        g, gc = gear_for_speed_profiles(
            self.gs, self._velocity, self._gear, self._gear_count
        )

        a = correct_acc_clutch_on(gc, accMFC(
            self._velocity, self.driver_style, desired_velocity, self.curves[g - 1]
        ), self.transmission)
        v = self._velocity + a * dt
        s = self.position + self._velocity * dt + 0.5 * a * dt ** 2
        if update:
            self._gear, self._gear_count, self._velocity, self.position = g, gc, v, s
        return g, v, a, s
