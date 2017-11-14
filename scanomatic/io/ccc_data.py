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
from types import StringTypes
from enum import Enum
from uuid import uuid1
import os
import re
import json
from glob import iglob

from scanomatic.io.logger import Logger
from scanomatic.io.paths import Paths

_logger = Logger("CCC-data")


class CCCFormatError(Exception):
    pass


class CellCountCalibration(Enum):

    status = 0
    """:type : CellCountCalibration"""
    species = 1
    """:type : CellCountCalibration"""
    reference = 2
    """:type : CellCountCalibration"""
    identifier = 3
    """:type : CellCountCalibration"""
    images = 4
    """:type : CellCountCalibration"""
    polynomial = 6
    """:type : CellCountCalibration"""
    edit_access_token = 7
    """:type : CellCountCalibration"""


class CCCImage(Enum):

    identifier = 0
    """:type : CCCImage"""
    plates = 1
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
    compressed_ccc_data = 2
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
        CellCountCalibration.images: [],
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


def _get_new_image_identifier(ccc):

    return "CalibIm_{0}".format(len(ccc[CellCountCalibration.images]))


def get_empty_image_entry(ccc):

    return {
        CCCImage.identifier: _get_new_image_identifier(ccc),
        CCCImage.plates: {},
        CCCImage.grayscale_name: None,
        CCCImage.grayscale_source_values: None,
        CCCImage.grayscale_target_values: None,
        CCCImage.marker_x: None,
        CCCImage.marker_y: None,
        CCCImage.fixture: None,
    }


def parse_ccc(data):

    data = _decode_dict(data)

    for ccc_data_type in CellCountCalibration:
        if ccc_data_type not in data:
            _logger.error(
                "Corrupt CCC-data, missing {0} in {1}".format(
                    ccc_data_type, data)
            )
            return None

    return data


__DECODABLE_ENUMS = {
    "CellCountCalibration": CellCountCalibration,
    "CalibrationEntryStatus": CalibrationEntryStatus,
    "CCCImage": CCCImage,
    "CCCMeasurement": CCCMeasurement,
    "CCCPlate": CCCPlate,
    "CCCPolynomial": CCCPolynomial,
}


def _decode_ccc_key(key):
    if isinstance(key, StringTypes):
        if '.' in key:
            return _decode_ccc_enum(key)
        else:
            if re.match(r'\(\d+, \d+\)', key):
                return eval(key)
            else:
                try:
                    return int(key)
                except ValueError:
                    raise CCCFormatError("{} not recognized".format(key))
    raise CCCFormatError("{} not recognized".format(key))


def _decode_ccc_enum(value):
    enum_name, enum_value = value.split(".")
    try:
        return __DECODABLE_ENUMS[enum_name][enum_value]
    except (ValueError, KeyError):
        raise CCCFormatError("'{}' not a valid property of '{}'".format(
            enum_value, enum_name))


def _decode_val(v):
    if isinstance(v, dict):
        return _decode_dict(v)
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(_decode_val(e) for e in v)
    elif isinstance(v, StringTypes):
        try:
            return _decode_ccc_enum(v)
        except (CCCFormatError, ValueError):
            return v
    else:
        return v


def _decode_dict(d):
    return {_decode_ccc_key(k): _decode_val(v) for k, v in d.iteritems()}


def _encode_ccc_key(key):
    if isinstance(key, tuple):
        return str(key)
    elif isinstance(key, Enum):
        return _encode_ccc_enum(key)
    return key


def _encode_ccc_enum(key):
    if type(key) in __DECODABLE_ENUMS.values():
        return "{}.{}".format(str(key).split(".")[-2], key.name)
    else:
        raise CCCFormatError("'{}' not usable in CCC keys".format(key))


def _encode_val(v):
    if isinstance(v, dict):
        return _encode_dict(v)
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(_encode_val(e) for e in v)
    else:
        try:
            return _encode_ccc_enum(v)
        except CCCFormatError:
            return v


def _encode_dict(d):
    return {_encode_ccc_key(k): _encode_val(v) for k, v in d.iteritems()}


def save_ccc(data):

    identifier = data[CellCountCalibration.identifier]

    try:
        os.makedirs(
            os.path.dirname(Paths().ccc_file_pattern.format(identifier)))
    except os.error:
        pass

    with open(Paths().ccc_file_pattern.format(identifier), 'wb') as fh:
        json.dump(_encode_dict(data), fh)

    return True


def load_cccs():

    for ccc_path in iglob(Paths().ccc_file_pattern.format("*")):

        with open(ccc_path, mode='rb') as fh:
            try:
                data = json.load(fh)
            except ValueError:
                _logger.warning("JSON format corrupt in file '{}'".format(
                    ccc_path
                ))
                continue
        try:
            data = parse_ccc(data)
        except CCCFormatError:
            _logger.warning("{} is an outdated CCC".format(ccc_path))

        if (data is None or
                CellCountCalibration.identifier not in data or
                not data[CellCountCalibration.identifier]):

            _logger.error("Data file '{0}' is corrupt.".format(ccc_path))

        else:
            yield data
