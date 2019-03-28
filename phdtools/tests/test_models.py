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
