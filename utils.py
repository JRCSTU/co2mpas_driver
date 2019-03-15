#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
It contains classes and functions of general utility.
These are python-specific utilities and hacks - general data-processing or
numerical operations.
"""


from collections import OrderedDict, defaultdict
from contextlib import contextmanager
import inspect
import io
import re
import statistics
import sys
import yaml

import networkx as nx
import numpy as np


try:
    isidentifier = str.isidentifier
except AttributeError:
    isidentifier = re.compile(r'[a-z_]\w*$', re.I).match

__all__ = [
    'grouper', 'sliding_window', 'median_filter', 'reject_outliers',
    'clear_fluctuations', 'argmax', 'derivative'
]


class Constants(dict):
    @nx.utils.open_file(1, mode='rb')
    def load(self, file, **kw):
        self.from_dict(yaml.load(file, **kw))
        return self

    @nx.utils.open_file(1, mode='w')
    def dump(self, file, default_flow_style=False, **kw):
        d = self.to_dict()
        yaml.dump(d, file, default_flow_style=default_flow_style, **kw)

    def from_dict(self, d):
        for k, v in sorted(d.items()):
            if isinstance(v, Constants):
                o = getattr(self, k, Constants())
                if isinstance(o, Constants):
                    v = o.from_dict(v)
                elif issubclass(o.__class__, Constants):
                    v = o().from_dict(v)
                if not v:
                    continue
            elif hasattr(self, k) and getattr(self, k) == v:
                continue
            setattr(self, k, v)
            self[k] = v

        return self

    def to_dict(self, base=None):
        pr = {} if base is None else base
        s = (set(dir(self)) - set(dir(Constants)))
        for n in s.union(self.__class__.__dict__.keys()):
            if n.startswith('__'):
                continue
            v = getattr(self, n)
            if inspect.ismethod(v) or inspect.isbuiltin(v):
                continue
            try:
                if isinstance(v, Constants):
                    v = v.to_dict(base=Constants())
                elif issubclass(v, Constants):
                    v = v.to_dict(v, base=Constants())
            except TypeError:
                pass
            pr[n] = v
        return pr


def argmax(values, **kws):
    return np.argmax(np.append(values, [True]), **kws)


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks.

    :param iterable:
        Iterable object.
    :param iterable: iter

    :param n:
        Length chunks or blocks.
    :type n: int
    """
    args = [iter(iterable)] * n
    return zip(*args)


def sliding_window(xy, dx_window):
    """
    Returns a sliding window (of width dx) over data from the iterable.

    :param xy:
        X and Y values.
    :type xy: list[(float, float) | list[float]]

    :param dx_window:
        dX window.
    :type dx_window: float

    :return:
        Data (x & y) inside the time window.
    :rtype: generator
    """

    dx = dx_window / 2
    it = iter(xy)
    v = next(it)
    window = []

    for x, y in xy:
        # window limits
        x_dn = x - dx
        x_up = x + dx

        # remove samples
        window = [w for w in window if w[0] >= x_dn]

        # add samples
        while v and v[0] <= x_up:
            window.append(v)
            try:
                v = next(it)
            except StopIteration:
                v = None

        yield window


def median_filter(x, y, dx_window, filter=statistics.median_high):
    """
    Calculates the moving median-high of y values over a constant dx.

    :param x:
        x data.
    :type x: Iterable

    :param y:
        y data.
    :type y: Iterable

    :param dx_window:
        dx window.
    :type dx_window: float

    :param filter:
        Filter function.
    :type filter: function

    :return:
        Moving median-high of y values over a constant dx.
    :rtype: numpy.array
    """

    xy = list(zip(x, y))
    Y = []
    add = Y.append
    for v in sliding_window(xy, dx_window):
        add(filter(list(zip(*v))[1]))
    return np.array(Y)


def get_inliers(x, n=1, med=np.median, std=np.std):
    x = np.asarray(x)
    if not x.size:
        return np.zeros_like(x, dtype=bool), np.nan, np.nan
    m, s = med(x), std(x)

    y = n > (np.abs(x - m) / s)
    return y, m, s


def reject_outliers(x, n=1, med=np.median, std=np.std):
    """
    Calculates the median and standard deviation of the sample rejecting the
    outliers.

    :param x:
        Input data.
    :type x: Iterable

    :param n:
        Number of standard deviations.
    :type n: int

    :param med:
        Median function.
    :type med: function, optional

    :param std:
        Standard deviation function.
    :type std: function, optional

    :return:
        Median and standard deviation.
    :rtype: (float, float)
    """

    y, m, s = get_inliers(x, n=n, med=med, std=std)

    if y.any():
        y = np.asarray(x)[y]

        m, s = med(y), std(y)

    return m, s


def ret_v(v):
    """
    Returns a function that return the argument v.

    :param v:
        Object to be returned.
    :type v: object

    :return:
        Function that return the argument v.
    :rtype: function
    """

    return lambda: v


def clear_fluctuations(times, gears, dt_window):
    """
    Clears the gear identification fluctuations.

    :param times:
        Time vector.
    :type times: numpy.array

    :param gears:
        Gear vector.
    :type gears: numpy.array

    :param dt_window:
        Time window.
    :type dt_window: float

    :return:
        Gear vector corrected from fluctuations.
    :rtype: numpy.array
    """

    xy = [list(v) for v in zip(times, gears)]

    for samples in sliding_window(xy, dt_window):

        up, dn = False, False

        x, y = zip(*samples)

        for k, d in enumerate(np.diff(y)):
            if d > 0:
                up = True
            elif d < 0:
                dn = True

            if up and dn:
                m = statistics.median_high(y)
                for v in samples:
                    v[1] = m
                break

    return np.array([y[1] for y in xy])


def _err(v, y1, y2, r, l):
    import sklearn.metrics as sk_met

    return sk_met.mean_absolute_error(_ys(y1, v) + _ys(y2, l - v), r)


def _ys(y, n):
    if n:
        return (y,) * int(n)
    return ()


def derivative(x, y, dx=1, order=3, k=1):
    """
    Find the 1-st derivative of a spline at a point.

    Given a function, use a central difference formula with spacing `dx` to
    compute the `n`-th derivative at `x0`.

    :param x:
    :param y:
    :param dx:
    :param order:
    :param k:
    :return:
    """
    import scipy.misc as sci_misc
    import scipy.interpolate as sci_itp

    func = sci_itp.InterpolatedUnivariateSpline(x, y, k=k)

    return sci_misc.derivative(func, x, dx=dx, order=order)


@contextmanager
def stds_redirected(stdout=None, stderr=None):
    captured_out = io.StringIO() if stdout is None else stdout
    captured_err = io.StringIO() if stderr is None else stderr
    orig_out, sys.stdout = sys.stdout, captured_out
    orig_err, sys.stderr = sys.stderr, captured_err

    yield captured_out, captured_err

    sys.stdout, sys.stderr = orig_out, orig_err


_key_value_regex = re.compile(r'^\s*([/_A-Za-z][\w/\.]*)\s*([+*?:@]?)=\s*(.*?)\s*$')



##############################
## Maintain ordered YAML
#  from http://stackoverflow.com/a/21912744
#
_MAPTAG = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG


def _construct_ordered_dict(loader, node):
    loader.flatten_mapping(node)
    return OrderedDict(loader.construct_pairs(node))


def _ordered_dict_representer(dumper, data):
    return dumper.represent_mapping(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        data.items())


def yaml_load(stream, Loader=yaml.SafeLoader):
    class OrderedLoader(Loader):
        pass

    OrderedLoader.add_constructor(_MAPTAG, _construct_ordered_dict)
    return yaml.load(stream, OrderedLoader)


def yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    OrderedDumper.add_representer(OrderedDict, _ordered_dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def setup_yaml_ordered():
    """
    Invoke it once it to enable app-wide ordered yaml.

    From http://stackoverflow.com/a/8661021 """
    yaml.add_representer(OrderedDict, _ordered_dict_representer)
    yaml.add_representer(defaultdict, _ordered_dict_representer)
    yaml.add_representer(tuple, yaml.SafeDumper.represent_list)
    yaml.add_constructor(_MAPTAG, _construct_ordered_dict)
