from __future__ import absolute_import

import numpy as np

from scanomatic.generics.maths import quantiles_stable


def test_quantiles_stable():
    data = np.arange(100)
    assert quantiles_stable(data) == (25, 75)
