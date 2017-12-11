import pytest
import json
from types import StringTypes

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


def test_parsing_ccc_string():

    data = """{
    "CellCountCalibration.identifier": 4,
    "CellCountCalibration.reference": 1,
    "CellCountCalibration.images": [],
    "CellCountCalibration.edit_access_token": "leet",
    "CellCountCalibration.polynomial": null,
    "CellCountCalibration.status": "CalibrationEntryStatus.UnderConstruction",
    "(1, 3)": "test",
    "4": [
        {
            "CCCImage.identifier": "hello to you",
            "CCCPolynomial.power": "CCCPolynomial.coefficients"
        }
    ],
    "CellCountCalibration.species": [1, 2, 3, 4, 5]
}"""

    assert ccc_data.parse_ccc(json.loads(data)) == {
        ccc_data.CellCountCalibration.identifier: 4,
        ccc_data.CellCountCalibration.reference: 1,
        ccc_data.CellCountCalibration.images: [],
        ccc_data.CellCountCalibration.edit_access_token: 'leet',
        ccc_data.CellCountCalibration.polynomial: None,
        ccc_data.CellCountCalibration.status:
        ccc_data.CalibrationEntryStatus.UnderConstruction,
        (1, 3): "test",
        4: [
            {
                ccc_data.CCCImage.identifier: 'hello to you',
                ccc_data.CCCPolynomial.power:
                ccc_data.CCCPolynomial.coefficients
            }
        ],
        ccc_data.CellCountCalibration.species: [1, 2, 3, 4, 5]

    }


def test_load_cccs():

    for ccc in ccc_data.load_cccs():

        assert isinstance(
            ccc[ccc_data.CellCountCalibration.identifier], StringTypes)
        assert ccc[ccc_data.CellCountCalibration.edit_access_token]
        assert ccc[ccc_data.CellCountCalibration.species]
        assert ccc[ccc_data.CellCountCalibration.reference]
        assert isinstance(ccc[ccc_data.CellCountCalibration.images], list)
        assert isinstance(
            ccc[ccc_data.CellCountCalibration.status],
            ccc_data.CalibrationEntryStatus
        )
        assert (
            ccc[ccc_data.CellCountCalibration.polynomial] is None or
            len(ccc[ccc_data.CellCountCalibration.polynomial][
                ccc_data.CCCPolynomial.coefficients]) ==
            ccc[ccc_data.CellCountCalibration.polynomial][
                ccc_data.CCCPolynomial.power] + 1
        )
