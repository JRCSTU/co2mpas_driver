import numpy.testing as nt


def _assert(v, r):
    if isinstance(v, str):
        nt.assert_equal(v, r)
    elif isinstance(v, dict):
        for k, v in v.items():
            _assert(v, r[k])
    elif isinstance(v, list):
        for v, r in zip(v, r):
            _assert(v, r)
    else:
        nt.assert_almost_equal(v, r)


def test_check(dsp, inputs, outputs):
    res = dsp(inputs, outputs)
    nt.assert_equal(
        set(outputs).issubset(res), True,
        "Missing outputs {}".format(set(outputs) - set(res))
    )
    for k, v in outputs.items():
        _assert(v, res[k])
