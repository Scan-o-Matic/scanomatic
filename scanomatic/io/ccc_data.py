"""CCC-data io.

Data structure for CCC-jsons
{
    CellCountCalibration.status:
        CalibrationEntryStatus,  # One of UnderConstruction, Active, Deleted
    CellCountCalibration.edit_access_token:
        string, # During CalibrationEntryStatus.UnderConstruction this is
                # needed to edit.
    CellCountCalibration.species:
        string,  # The species & possibly strain, combo of this and reference
                 # must be unique.
    CellCountCalibration.reference:
        string,  # Typically publication reference or contact info i.e. email
    CellCountCalibration.identifier:
        string,  # Unique ID of CCC
    CellCountCalibration.images:
        list,  # The images in sequence added, must correspond to order
               # in reference data (below)
        [
            {
            CCCImage.identifier: string,  # How to find the saved image
            CCCImage.plates:
                {
                int :  # plate index (get valid from fixture),
                    {
                    CCCPlate.grid_shape: (16, 24),
                        # Number of rows and columns of colonies on plate
                    CCCPlate.grid_cell_size: (52.5, 53.1),
                        # Number of pixels for each colony (yes is in decimal)
                    CCCPlate.compressed_ccc_data:
                        # Row, column info on CCC analysis of each colony
                        [
                            [
                                {
                                    CCCMeasurement.source_values:
                                        [123.1, 10.4, ...],
                                        # GS transf pixel transparencies
                                    CCCMeasurement.source_value_counts:
                                        [100, 1214, ...],
                                        # Num of corresponding pixels
                                    CCCMeasurement.cell_count: 300000
                                        # Num of cells (independent data)
                                },
                                ...
                            ],
                            ...
                        ],
                    }
                },
            CCCImage.grayscale_name: string,
            CCCImage.grayscale_source_values:
                [123, 2.14, ...],  # reference values
            CCCImage.grayscale_target_values:
                [123, 12412, ...], # Analysis values
            CCCImage.fixture: string,
            },
            ...
        ],
    CellCountCalibration.polynomial:
        {
            CCCPolynomial.power: int,
            CCCPolynomial.coefficients: [10, 0, 0, 0, 150, 0]
        },
        # Or None
    }
}

"""
from __future__ import absolute_import

from enum import Enum
from uuid import uuid1

from scanomatic.io.logger import Logger

_logger = Logger("CCC-data")


class CellCountCalibration(Enum):

    status = 0
    """:type : CellCountCalibration"""
    species = 1
    """:type : CellCountCalibration"""
    reference = 2
    """:type : CellCountCalibration"""
    identifier = 3
    """:type : CellCountCalibration"""
    polynomial = 6
    """:type : CellCountCalibration"""
    edit_access_token = 7
    """:type : CellCountCalibration"""


class CCCImage(Enum):

    identifier = 0
    """:type : CCCImage"""
    grayscale_name = 2
    """:type : CCCImage"""
    grayscale_source_values = 3
    """:type : CCCImage"""
    grayscale_target_values = 4
    """:type : CCCImage"""
    fixture = 5
    """:type : CCCImage"""
    marker_x = 6
    """:type : CCCImage"""
    marker_y = 7
    """:type : CCCImage"""


class CCCPlate(Enum):
    grid_shape = 0
    """:type : CCCPlate"""
    grid_cell_size = 1
    """:type : CCCPlate"""


class CCCMeasurement(Enum):
    source_values = 1
    """:type : CCCMeasurement"""
    source_value_counts = 2
    """:type : CCCMeasurement"""
    cell_count = 3
    """:type : CCCMeasurement"""


class CCCPolynomial(Enum):
    power = 0
    coefficients = 1


class CalibrationEntryStatus(Enum):

    UnderConstruction = 0
    """:type: CalibrationEntryStatus"""
    Active = 1
    """:type: CalibrationEntryStatus"""
    Deleted = 2
    """:type: CalibrationEntryStatus"""


def get_empty_ccc_entry(ccc_id, species, reference):
    return {
        CellCountCalibration.identifier: ccc_id,
        CellCountCalibration.species: species,
        CellCountCalibration.reference: reference,
        CellCountCalibration.edit_access_token: uuid1().hex,
        CellCountCalibration.polynomial: None,
        CellCountCalibration.status: CalibrationEntryStatus.UnderConstruction,
    }


def get_polynomal_entry(power, poly_coeffs):

    return {
        CCCPolynomial.power: power,
        CCCPolynomial.coefficients: poly_coeffs,
    }


def validate_polynomial_format(polynomial):
    try:
        if (not (
                isinstance(polynomial[CCCPolynomial.power], int) and
                isinstance(polynomial[CCCPolynomial.coefficients], list) and
                len(polynomial[CCCPolynomial.coefficients]) ==
                polynomial[CCCPolynomial.power] + 1)):
            _logger.error(
                "Validation of polynomial representaiton {} failed".format(
                    polynomial)
            )
            raise ValueError(
                "Invalid polynomial representation: {}".format(polynomial)
            )
    except (KeyError, TypeError) as err:
        _logger.error(
            "Validation of polynomial representation failed with {}".format(
                err.message)
        )
        raise ValueError(err.message)


def get_empty_image_entry(identifier):

    return {
        CCCImage.identifier: identifier,
        CCCImage.grayscale_name: None,
        CCCImage.grayscale_source_values: None,
        CCCImage.grayscale_target_values: None,
        CCCImage.marker_x: None,
        CCCImage.marker_y: None,
        CCCImage.fixture: None,
    }
