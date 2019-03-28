from tools.dyntools import *
import pytest
import numpy as np

@pytest.mark.parametrize("input, exp_output", [
    (0.1, 1),
    (-0.1, -1),
    (-2, -1),
    (2, 1)
])
def test_stabil(input, exp_output):

    tol = 1e-4

    def model(y, t):
        ''' Model with equilibria at:
        -1 (stable)
        0  (unstable)
        1  (stable)
        '''
        dydx = -(y - 1)*y*(y + 1)
        return dydx

    y0 = input
    tStabil = (0, 1e2)
    yStabil = stabil(model, y0, tStabil)

    assert(yStabil == pytest.approx(exp_output, tol)), \
        'stabil is not behaving as expected'
