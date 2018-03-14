from __future__ import absolute_import

import pytest

from scanomatic.io import ccc_data


class TestPolynomials:
    @pytest.mark.parametrize("polynomial", [
        {
            ccc_data.CCCPolynomial.power: "apa",
            ccc_data.CCCPolynomial.coefficients: [0, 1]
        },
        {
            ccc_data.CCCPolynomial.power: 1.0,
            ccc_data.CCCPolynomial.coefficients: [0, 1]
        },
        {
            ccc_data.CCCPolynomial.power: 2,
            ccc_data.CCCPolynomial.coefficients: [0, 1]
        },
        {"browser": 2, "coffee": [0, 1]},
        {'power': 1, 'coefficients': [0, 1]},
    ])
    def test_polynomial_malformed(self, polynomial):
        with pytest.raises(ValueError):
            ccc_data.validate_polynomial_format(polynomial)

    @pytest.mark.parametrize("polynomial", [
        {
            ccc_data.CCCPolynomial.power: 0,
            ccc_data.CCCPolynomial.coefficients: [0]
        },
        {
            ccc_data.CCCPolynomial.power: 1,
            ccc_data.CCCPolynomial.coefficients: [0, 1]
        },
        {
            ccc_data.CCCPolynomial.power: 2,
            ccc_data.CCCPolynomial.coefficients: [1, 2, 3]
        },
    ])
    def test_polynomial_correct(self, polynomial):
        assert ccc_data.validate_polynomial_format(polynomial) is None
