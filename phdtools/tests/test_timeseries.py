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

@pytest.mark.parametrize("Dt, i, width, exp_output", [
    ((1,2,3,4,5), 3, 2, (3,4))
])
def test_window_discrete(Dt, i, width, exp_output):

    Dt_subsetted = window_discrete(Dt, i, width)
    assert(len(Dt_subsetted) == width)
    assert(Dt_subsetted == exp_output)

@pytest.mark.parametrize("Dt, i, width", [
    ((1,2,3,4,5), 1, 3), # Too wide
    ((1,2,3,4,5), 10, 1) # Too a high index
])
def test_window_discrete_nan(Dt, i, width):

    Dt_subsetted = window_discrete(Dt, i, width)
    assert(np.isnan(Dt_subsetted))

@pytest.mark.parametrize("Dt, width", [
    ((1,1,1,1,3), 2)
])
def test_mean_window_discrete(Dt, width):

    s = mean_window_discrete(Dt, width)
    s_expected = np.array([np.nan, 1.0, 1.0, 1.0, 2.0])
    for i in range(0, len(s)):
        if i == 0:
            assert(np.isnan(s[i]))
        else:
            assert(s[i] == s_expected[i])

@pytest.mark.parametrize("Dt, width", [
    ((1,1,1,1,1), 2)
])
def test_std_window_discrete(Dt, width):

    s = std_window_discrete(Dt, width)
    s_expected = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])
    for i in range(0, len(s)):
        if i == 0:
            assert(np.isnan(s[i]))
        else:
            assert(s[i] == s_expected[i])

@pytest.mark.parametrize("Dt, width", [
    ((1,1,1,1,1), 2)
])
def test_var_window_discrete(Dt, width):

    s = var_window_discrete(Dt, width)
    s_expected = np.array([np.nan, 0.0, 0.0, 0.0, 0.0])
    for i in range(0, len(s)):
        if i == 0:
            assert(np.isnan(s[i]))
        else:
            assert(s[i] == s_expected[i])

def test_plot_autocorrelation():

    ts = range(0, 100)
    Dt = np.sin(ts)

    fig, ax = plt.subplots(1,1)
    ax = plot_autocorrelation(ax, Dt, range(0, 50))

def test_plot_return():

    ts = np.linspace(0, 20, 100)
    Dt = np.sin(ts)

    fig, ax = plt.subplots(1,1)
    ax = plot_return(ax, Dt, 2)

def test_plot_approx_phas():

    ts = np.linspace(0, 20, 100)
    Dt = np.sin(ts)

    fig, ax = plt.subplots(1,1)
    ax = plot_approx_phas(ax, Dt, ts)
