from phdtools.timeseries import *
import pytest
import numpy as np


@pytest.mark.parametrize("input, k, exp_output1, exp_output2", [
    ((1,2,3), 1, (1,2), (2,3)),
    ((4,5,6), 0, (4,5,6), (4,5,6))
])
def test_symm_subset(input, k, exp_output1, exp_output2):

    Vt, Vt_k = symm_subset(input, k)

    assert(Vt == exp_output1)
    assert(Vt_k == exp_output2)

def test_fluctuations():

    ts = np.linspace(0, 20, 100)
    xt = 10 + np.sin(ts)

    Vt = fluctuations(xt)

    tol = 1e-2
    assert(np.mean(Vt) == pytest.approx(0.0, tol))

def test_measure():

    ts = np.linspace(0, 20, 100)
    xt = 10 + np.sin(ts)

    # Test without sampling error
    Dt = measure(xt, scale=0.0)
    assert((Dt == xt).all())

    # Test with sampling error
    Dt = measure(xt, scale=0.25)
    assert((Dt == xt).all() == False) # Not exactly...
    tol = 1 # ... but approximately...
    assert((Dt == pytest.approx(xt, tol))) #... correct

@pytest.mark.parametrize("Dt, k, exp_output", [
    ((1,2,3,4,5), 0, 1),
    ((1,2,3,4,5), 1, 0.6666666)
])
def test_autocorrelation(Dt, k, exp_output):

    ac = autocorrelation(Dt, k)

    tol = 1e-4
    assert(ac == pytest.approx(exp_output, tol))
