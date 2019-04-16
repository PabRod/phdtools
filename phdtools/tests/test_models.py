from phdtools.models import *
import pytest
import numpy as np


@pytest.mark.parametrize("input, l, exp_output", [
    (4, 1, -4),
    (-4, 2, 8),
    (0, 1, 0)
])
def test_decay(input, l, exp_output):

    tol = 1e-8

    df = decay(input, 0, l)

    assert(df == pytest.approx(exp_output, tol)), \
        'decay is not behaving as expected'

@pytest.mark.parametrize("input, exp_output", [
    (0,0),
    (1,0),
    (0.5, 0.375)
])
def test_ass(input, exp_output):

    tol = 1e-8

    df = ass(input)

    assert(df == pytest.approx(exp_output, tol)), \
        'ass is not behaving as expected'


@pytest.mark.parametrize("input, exp_output, a, b, c, d", [
    ((1, 1), (0, 0), 1, 1, 1, 1)
])
def test_lotkavolterra(input, exp_output, a, b, c, d):

    tol = 1e-8

    df = lotkavolterra(input, 0, a, b, c, d)

    assert(df == pytest.approx(exp_output, tol)), \
        'lotkavolterra is not behaving as expected'

@pytest.mark.parametrize("input, exp_output", [
    ((0, 0), (0, 0)),
    ((10, 0), (0, 0)),
    ((3.333333333333294, 4.444444444444456), (0, 0)),
    ((1, 1), (19/60, -7/100))
])
def test_rosmac(input, exp_output):

    tol = 1e-8

    df = rosmac(input)

    assert(df == pytest.approx(exp_output, tol)), \
        'rosmac is not behaving as expected'
