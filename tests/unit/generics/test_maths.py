import numpy as np

from scanomatic.generics.maths import quantiles_stable

def test_quantiles_stable():
    data = np.array(range(100))
    assert quantiles_stable(data) == (25, 75)
