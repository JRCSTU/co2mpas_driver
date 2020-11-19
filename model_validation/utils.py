import schedula as sh
import numpy.testing as nt


def _assert(v, r, name):
    if isinstance(v, str):
        nt.assert_equal(v, r, 'Value %s is not equal!' % name)
    elif isinstance(v, dict):
        for k, v in v.items():
            _assert(v, r[k], '{}/{}'.format(name, k))
    elif isinstance(v, list):
        for i, (v, r) in enumerate(zip(v, r)):
            _assert(v, r, '{}/{}'.format(name, i))
    elif callable(v):
        pass
    else:
        nt.assert_almost_equal(v, r, err_msg='Value %s is not equal!' % name)


def _check(dsp, data, outputs):
    inputs = sh.selector(set(data) - set(outputs), data)
    res = dsp(inputs, outputs)
    for d in res.workflow.nodes.values():
        if 'solution' not in d:
            continue
        s = d['solution']
        nt.assert_equal(
            bool(s._errors), False, "Found errors in {}".format(set(s._errors))
        )
    nt.assert_equal(
        set(outputs).issubset(res), True,
        "Missing outputs {}".format(set(outputs) - set(res))
    )
    for k in outputs:
        _assert(data[k], res[k], k)
