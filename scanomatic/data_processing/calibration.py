import numpy as np
from enum import Enum
import json
from itertools import izip
import os
import shutil
from scipy.optimize import curve_fit
import time
from datetime import datetime
from dateutil import tz
from scipy.stats import linregress
import re
from uuid import uuid1
from glob import iglob
from types import StringTypes
from collections import namedtuple


from scanomatic.generics.maths import mid50_mean
from scanomatic.io.logger import Logger
from scanomatic.io.paths import Paths
from scanomatic.io.fixtures import Fixtures
from scanomatic.io.meta_data import MetaData2 as MetaData
from scanomatic.image_analysis.image_basics import (
    load_image_to_numpy, Image_Transpose)
from scanomatic.image_analysis.first_pass_image import FixtureImage

""" Data structure for CCC-jsons
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
                                    CCCMeasurement.included: bool,
                                        # If included
                                    CCCMeasurement.source_values:
                                        [123.1, 10.4, ...],
                                        # GS transf pixel transparencies
                                    CCCMeasurement.source_value_counts:
                                        [100, 1214, ...],
                                        # Num of corresponding pixels
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
    CellCountCalibration.independent_data:
        [[[12300, 121258, 1241240, 141410, ...], ..., ...]
        # Continuous list of measurements of population sizes from OD or FACS
        # Plates, columns, row
    CellCountCalibration.independent_data_source: string  # File format
    CellCountCalibration.polynomial: {
        string: {'power': int, 'coefficients': [10, 0, 0, 0, 150, 0]},
        ....
    }
    CellCountCalibration.deployed_polynomial: string
}

"""

__CCC = {}
_logger = Logger("Cell Count Calibration")


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
    independent_data = 5
    """:type : CellCountCalibration"""
    polynomial = 6
    """:type : CellCountCalibration"""
    edit_access_token = 7
    """:type : CellCountCalibration"""
    independent_data_source = 8
    """:type : CellCountCalibration"""
    deployed_polynomial = 9
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
    included = 0
    """:type : CCCMeasurement"""
    source_values = 1
    """:type : CCCMeasurement"""
    source_value_counts = 2
    """:type : CCCMeasurement"""


CalibrationEntry = namedtuple(
    "CalibrationEntry", [
        'image',
        'colony_name',
        'target_value',
        'source_values',
        'source_value_counts'
    ]
)


class ActivationError(Exception):
    pass


class CalibrationEntryStatus(Enum):

    UnderConstruction = 0
    """:type: CalibrationEntryStatus"""
    Active = 1
    """:type: CalibrationEntryStatus"""
    Deleted = 2
    """:type: CalibrationEntryStatus"""


class CalibrationValidation(Enum):
    OK = 0
    """:type : CalibrationValidation"""
    BadSlope = 1
    """:type : CalibrationValidation"""
    BadStatistics = 3
    """:type : CalibrationValidation"""


def _validate_ccc_edit_request(f):

    def wrapped(identifier, *args, **kwargs):

        if identifier in __CCC:

            ccc = __CCC[identifier]

            if ("access_token" in kwargs and
                    ccc[CellCountCalibration.edit_access_token] ==
                    kwargs["access_token"] and
                    kwargs["access_token"]):

                if (ccc[CellCountCalibration.status] ==
                        CalibrationEntryStatus.UnderConstruction):

                    del kwargs["access_token"]
                    return f(identifier, *args, **kwargs)

                else:

                    _logger.error(
                        "Can not modify the CCC {0} because it is not under construction".format(
                            identifier)
                    )

            else:

                _logger.error(
                    "You don't have the correct access token for {0}, request refused".format(identifier)
                )
        else:

            _logger.error("Unknown CCC {0}".format(identifier))

    return wrapped


@_validate_ccc_edit_request
def is_valid_edit_request(identifier):
    return True


def get_empty_ccc(species, reference):
    ccc_id = _get_ccc_identifier(species, reference)
    if ccc_id is None:
        return None

    return {
        CellCountCalibration.identifier: ccc_id,
        CellCountCalibration.species: species,
        CellCountCalibration.reference: reference,
        CellCountCalibration.images: [],
        CellCountCalibration.edit_access_token: uuid1().hex,
        CellCountCalibration.polynomial: None,
        CellCountCalibration.status: CalibrationEntryStatus.UnderConstruction,
        CellCountCalibration.independent_data: [],
        CellCountCalibration.independent_data_source: None,
        CellCountCalibration.deployed_polynomial: None,
    }


def _get_ccc_identifier(species, reference):

    if not species or not reference:
        return None

    if any(True for ccc in __CCC.itervalues() if
           ccc[CellCountCalibration.species] == species and
           ccc[CellCountCalibration.reference] == reference):

        return None

    candidate = re.sub(r'[^A-Z]', r'', species.upper())[:6]

    while any(True for ccc in __CCC.itervalues()
              if ccc[CellCountCalibration.identifier] == candidate):

        candidate += "qwxz"[np.random.randint(0, 4)]

    return candidate


def __load_cccs():

    for ccc_path in iglob(Paths().ccc_file_pattern.format("*")):

        with open(ccc_path, mode='rb') as fh:
            data = json.load(fh)

        data = _parse_ccc(data)

        if (data is None or
                CellCountCalibration.identifier not in data or
                not data[CellCountCalibration.identifier]):

            _logger.error("Data file '{0}' is corrupt.".format(ccc_path))

        elif data[CellCountCalibration.identifier] in __CCC:

            _logger.error(
                "Duplicated identifier {0} is not allowed!".format(
                    data[CellCountCalibration.identifier])
            )

        else:

            __CCC[data[CellCountCalibration.identifier]] = data


def get_active_cccs():

    return {
        identifier: ccc for identifier, ccc in __CCC.iteritems()
        if ccc[CellCountCalibration.status] == CalibrationEntryStatus.Active}


def get_under_construction_cccs():

    return {
        identifier: ccc for identifier, ccc in __CCC.iteritems()
        if ccc[CellCountCalibration.status] ==
        CalibrationEntryStatus.UnderConstruction}


def add_ccc(ccc):

    if (ccc[CellCountCalibration.identifier] and
            ccc[CellCountCalibration.identifier] not in __CCC):

        __CCC[ccc[CellCountCalibration.identifier]] = ccc
        save_ccc_to_disk(ccc[CellCountCalibration.identifier])
        return True

    else:

        _logger.error(
            "'{0}' is not a valid new CCC identifier".format(
                ccc[CellCountCalibration.identifier])
        )
        return False


def validate_polynomial_struct(polynomial):
    try:
        if (not (
                isinstance(polynomial['power'], int) and
                isinstance(polynomial['coefficients'], list) and
                len(polynomial['coefficients']) == polynomial['power'] + 1)):
            _logger.error(
                "Validation of polynomial representaiton {} failed".format(
                    polynomial)
            )
            raise ActivationError(
                "Invalid polynomial representation: {}".format(polynomial)
            )
    except (KeyError, TypeError) as err:
        _logger.error(
            "Validation of polynomial representation failed with {}".format(
                err.message)
        )
        raise ActivationError(err.message)


def has_valid_polynomial(ccc):
    try:
        validate_polynomial_struct(
            ccc[CellCountCalibration.polynomial][
                ccc[CellCountCalibration.deployed_polynomial]
            ]
        )
    except (KeyError, TypeError) as err:
        _logger.error(
            "Checking that CCC has valid polynomial failed with {}".format(
                err.message)
        )
        raise ActivationError(err.message)


@_validate_ccc_edit_request
def activate_ccc(identifier):

    ccc = __CCC[identifier]
    try:
        has_valid_polynomial(ccc)
    except ActivationError:
        return False

    ccc[CellCountCalibration.status] = CalibrationEntryStatus.Active
    ccc[CellCountCalibration.edit_access_token] = uuid1().hex

    return save_ccc_to_disk(identifier)


@_validate_ccc_edit_request
def delete_ccc(identifier):

    ccc = __CCC[identifier]

    ccc[CellCountCalibration.status] = CalibrationEntryStatus.Deleted
    ccc[CellCountCalibration.edit_access_token] = uuid1().hex

    return save_ccc_to_disk(identifier)


def save_ccc_to_disk(identifier):

    if identifier in __CCC:

        return _save_ccc_to_disk(__CCC[identifier])

    else:

        _logger.error("Unknown CCC identifier {0}".format(identifier))
        return False


def _encode_val(v):
    if isinstance(v, dict):
        return _encode_dict(v)
    if isinstance(v, list) or isinstance(v, tuple):
        return type(v)(_encode_val(e) for e in v)
    else:
        return _encode_ccc_enum(v)


def _encode_dict(d):
    return {_encode_ccc_enum(k): _encode_val(v) for k, v in d.iteritems()}


def _save_ccc_to_disk(data):

    identifier = data[CellCountCalibration.identifier]

    try:
        os.makedirs(
            os.path.dirname(Paths().ccc_file_pattern.format(identifier)))
    except os.error:
        pass

    with open(Paths().ccc_file_pattern.format(identifier), 'wb') as fh:
        json.dump(_encode_dict(data), fh)

    return True


def _parse_ccc(data):

    def _decode_val(v):
        if isinstance(v, dict):
            return _decode_dict(v)
        if isinstance(v, list) or isinstance(v, tuple):
            return type(v)(_decode_val(e) for e in v)
        else:
            return _decode_ccc_enum(v)

    def _decode_dict(d):
        return {_decode_ccc_enum(k): _decode_val(v) for k, v in d.iteritems()}

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
}


def _decode_ccc_enum(val):
    if isinstance(val, StringTypes):
        try:
            enum_name, enum_value = val.split(".")
            return __DECODABLE_ENUMS[enum_name][enum_value]
        except (ValueError, KeyError):
            pass
    return val


def _encode_ccc_enum(val):

    if type(val) in __DECODABLE_ENUMS.values():
        return "{0}.{1}".format(str(val).split(".")[-2], val.name)
    else:
        return val


@_validate_ccc_edit_request
def add_image_to_ccc(identifier, image):

    ccc = __CCC[identifier]
    im_json = _get_new_image_json(ccc)
    im_identifier = im_json[CCCImage.identifier]
    image.save(Paths().ccc_image_pattern.format(identifier, im_identifier))

    ccc[CellCountCalibration.images].append(im_json)
    if not save_ccc_to_disk(ccc):
        return False

    return im_identifier


def get_image_identifiers_in_ccc(identifier):

    if identifier in __CCC:

        ccc = __CCC[identifier]

        return [
            im_json[CCCImage.identifier] for im_json in
            ccc[CellCountCalibration.images]
        ]

    return False


@_validate_ccc_edit_request
def set_image_info(identifier, image_identifier, **kwargs):

    ccc = __CCC[identifier]
    im_json = get_image_json_from_ccc(identifier, image_identifier)

    for key in kwargs:

        try:

            im_json[CCCImage[key]] = kwargs[key]

        except (KeyError, TypeError):

            _logger.error("{0} is not a known property of images".format(key))
            return False

    return save_ccc_to_disk(ccc)


@_validate_ccc_edit_request
def set_plate_grid_info(
        identifier, image_identifier, plate,
        grid_shape=None, grid_cell_size=None, **kwargs):

    ccc = __CCC[identifier]
    im_json = get_image_json_from_ccc(identifier, image_identifier)
    if plate in im_json[CCCImage.plates]:
        plate_json = im_json[CCCImage.plates][plate]
        if plate_json[CCCPlate.compressed_ccc_data]:
            return False
    else:
        plate_json = {
            CCCPlate.grid_cell_size: None,
            CCCPlate.grid_shape: None, CCCPlate.compressed_ccc_data: []
        }
        im_json[CCCImage.plates][plate] = plate_json

    plate_json[CCCPlate.grid_cell_size] = grid_cell_size
    plate_json[CCCPlate.grid_shape] = grid_shape

    return save_ccc_to_disk(ccc)


def get_image_json_from_ccc(identifier, image_identifier):

    if identifier in __CCC:

        ccc = __CCC[identifier]

        for im_json in ccc[CellCountCalibration.images]:

            if im_json[CCCImage.identifier] == image_identifier:

                return im_json
    return None


def _get_new_image_json(ccc):

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


def _get_new_image_identifier(ccc):

    return "CalibIm_{0}".format(len(ccc[CellCountCalibration.images]))


def get_local_fixture_for_image(identifier, image_identifier):

    im_json = get_image_json_from_ccc(identifier, image_identifier)
    if im_json is None:
        return None

    fixture_settings = Fixtures()[im_json[CCCImage.fixture]]
    if fixture_settings is None:
        return None

    fixture = FixtureImage(fixture_settings)
    current_settings = fixture['current']
    current_settings.model.orientation_marks_x = np.array(
        im_json[CCCImage.marker_x])
    current_settings.model.orientation_marks_y = np.array(
        im_json[CCCImage.marker_y])
    issues = {}
    fixture.set_current_areas(issues)

    return dict(
        plates=current_settings.model.plates,
        grayscale=current_settings.model.grayscale,
        issues=issues,
    )


@_validate_ccc_edit_request
def save_image_slices(
        identifier, image_identifier, grayscale_slice=None, plate_slices=None):

    im = load_image_to_numpy(
        Paths().ccc_image_pattern.format(identifier, image_identifier),
        dtype=np.uint8)

    if grayscale_slice:
        np.save(
            Paths().ccc_image_gs_slice_pattern.format(
                identifier, image_identifier),
            _get_im_slice(im, grayscale_slice)
        )

    if plate_slices:
        for plate_model in plate_slices:
            np.save(
                Paths().ccc_image_plate_slice_pattern.format(
                    identifier, image_identifier, plate_model.index),
                _get_im_slice(im, plate_model)
            )

    return True


def _get_im_slice(im, model):

    return im[
        int(np.floor(model.y1)): int(np.ceil(model.y2)),
        int(np.floor(model.x1)): int(np.ceil(model.x2))
    ]


def get_grayscale_slice(identifier, image_identifier):

    try:
        return np.load(Paths().ccc_image_gs_slice_pattern.format(
            identifier, image_identifier)
        )
    except IOError:
        return None


def get_plate_slice(
        identifier, image_identifier, id_plate, gs_transformed=False):

    if gs_transformed:
        try:
            return np.load(
                Paths().ccc_image_plate_transformed_slice_pattern.format(
                    identifier, image_identifier, id_plate)
            )
        except IOError:
            _logger.error(
                "Problem loading: {0}".format(
                    Paths().ccc_image_plate_transformed_slice_pattern.format(
                    identifier, image_identifier, id_plate)
                )
            )
            return None
    else:
        try:
            return np.load(
                Paths().ccc_image_plate_slice_pattern.format(
                    identifier, image_identifier, id_plate)
            )
        except IOError:
            _logger.error(
                "Problem loading: {0}".format(
                    Paths().ccc_image_plate_slice_pattern.format(
                        identifier, image_identifier, id_plate)
                )
            )
            return None


@_validate_ccc_edit_request
def transform_plate_slice(identifier, image_identifier, plate_id):

    im_json = get_image_json_from_ccc(identifier, image_identifier)
    if not im_json:
        _logger.error(
            "CCC {0} Image {1} has not been setup, you must first add the image before working on it.".format(
                identifier, image_identifier)
        )

    plate = get_plate_slice(
        identifier, image_identifier, plate_id, gs_transformed=False)

    if plate is None:
        _logger.error(
            "No plate slice has been saved for {0}:{1}: plate {2}".format(
                identifier, image_identifier, plate_id)
        )
        return False

    grayscale_values = im_json[CCCImage.grayscale_source_values]
    grayscale_targets = im_json[CCCImage.grayscale_target_values]

    if not grayscale_targets or not grayscale_values:
        _logger.error("The gray-scale values have not been setup")

    transpose_polynomial = Image_Transpose(
        sourceValues=grayscale_values,
        targetValues=grayscale_targets)

    try:
        np.save(
            Paths().ccc_image_plate_transformed_slice_pattern.format(
                identifier, image_identifier, plate_id),
            transpose_polynomial(plate)
        )
        return True
    except IOError:
        _logger.error(
            "Problem saving: {0}".format(
                Paths().ccc_image_plate_transformed_slice_pattern.format(
                    identifier, image_identifier, plate_id)
            )
        )
        return False


@_validate_ccc_edit_request
def set_colony_compressed_data(
        identifier, image_identifier, plate_id, x, y, included=True,
        image=None, blob_filter=None, background_filter=None):

    ccc = __CCC[identifier]
    only_update_included = True
    if image is not None:
        only_update_included = False
        background = mid50_mean(image[background_filter].ravel())
        if np.isnan(background):
            _logger.error(
                "The background had too little information to make mid50 mean")
            return False

        colony = image[blob_filter].ravel() - background

        values, counts = zip(
            *{k: (colony == k).sum() for k in
              np.unique(colony).tolist()}.iteritems())

        if np.sum(counts) != blob_filter.sum():
            _logger.error(
                "Counting mismatch between compressed format and blob filter")
            return False

        image_data = get_image_json_from_ccc(identifier, image_identifier)
        plate = image_data[CCCImage.plates][plate_id]

    while len(plate[CCCPlate.compressed_ccc_data]) <= x:

        plate[CCCPlate.compressed_ccc_data].append([])

    while len(plate[CCCPlate.compressed_ccc_data][x]) <= y:

        plate[CCCPlate.compressed_ccc_data][x].append(
            {CCCMeasurement.included: False,
             CCCMeasurement.source_value_counts: [],
             CCCMeasurement.source_values: []})

    if only_update_included:

        if included and not (
                plate[CCCPlate.compressed_ccc_data][x][y][
                    CCCMeasurement.source_values] or
                plate[CCCPlate.compressed_ccc_data][x][y][
                    CCCMeasurement.source_value_counts]):

            _logger.warning(
                "Attempting to include CCC Measurement for position {0}, {1} while it has no data".format(
                    x, y)
            )

            return False

        plate[CCCPlate.compressed_ccc_data][x][y][
            CCCMeasurement.included] = included

    else:
        plate[CCCPlate.compressed_ccc_data][x][y][
            CCCMeasurement.included] = included
        plate[CCCPlate.compressed_ccc_data][x][y][
            CCCMeasurement.source_value_counts] = counts
        plate[CCCPlate.compressed_ccc_data][x][y][
            CCCMeasurement.source_values] = values

    return save_ccc_to_disk(ccc)

if not __CCC:
    __load_cccs()

########################################
########################################
########################################


def validate_polynomial(data, poly):

    expanded, targets, _, _ = _get_expanded_data(data)
    expanded_sums = np.array(tuple(v.sum() for v in poly(expanded)))
    slope, intercept, _, p_value, stderr = linregress(expanded_sums, targets)

    if abs(1.0 - slope) > 0.005:
        _logger.error("Bad slope for polynomial: {0}".format(slope))
        return CalibrationValidation.BadSlope

    if stderr > 0.05 or p_value > 0.1:
        return CalibrationValidation.BadStatistics

    return CalibrationValidation.OK


def _eval_deprecated_format(entry, key):

    if isinstance(key, CalibrationEntry):
        key = key.value

    if isinstance(key, int):
        return entry[key]
    elif key:
        return _eval_deprecated_format(entry[key[0]], key[1:])
    else:
        return entry


def _parse_data(entry):

    try:
        entry = json.loads(entry)
    except ValueError:
        # Try parsing old format
        entry = {k.name: _eval_deprecated_format(eval(entry), k)
                 for k in CalibrationEntry}

    return CalibrationEntry(**entry)


def _jsonify_entry(entry):

    return {k.name: v for k, v in entry.iteritems()}


def _jsonify(data):

    return json.dumps([_jsonify_entry(e) for e in data])


def get_data_file_path(file_path=None, label=''):
    if file_path is None:
        if label:
            file_path = Paths().analysis_calibration_data.format(label + ".")
        else:
            file_path = Paths().analysis_calibration_data.format(label)

    return file_path


def save_data_to_file(data, file_path=None, label=''):

    file_path = get_data_file_path(file_path, label)
    with open(file_path, 'w') as fh:
        json.dump(data, fh)


def load_data_file(file_path=None, label=''):

    file_path = get_data_file_path(file_path, label)
    try:
        with open(file_path, 'r') as fs:

            try:
                data_store = json.load(fs)

            except ValueError:

                data_store = CalibrationEntry(
                    target_value=[],
                    source_values=[],
                    source_value_counts=[],
                    image=None,
                    colony_name=None)

                for i, line in enumerate(fs):

                    try:
                        entry = _parse_data(line)
                    except (ValueError, TypeError):
                        entry = None

                    if entry:
                        data_store.source_value_counts.append(
                            entry.source_value_counts)
                        data_store.source_values.append(entry.source_values)
                        data_store.target_value.append(entry.target_value)

                    else:
                        _logger.warning(
                            "Could not parse line {0}: '{1}' in {2}".format(
                                i, line.strip(), file_path)
                        )

    except IOError:
        raise IOError("File at {0} not found".format(file_path))

    return data_store


def _collect_all_included_data(ccc):

    source_values = []
    source_value_counts = []
    target_value = []
    inclusion_filter = []

    for id_image, image_data in enumerate(ccc[CellCountCalibration.images]):

        for id_plate, plate in image_data[CCCImage.plates].items():

            for id_row, row in enumerate(plate[CCCPlate.compressed_ccc_data]):

                for id_col, item in enumerate(row):

                    try:
                        inclusion_filter.append(item[CCCMeasurement.included])

                        source_value_counts.append(
                            item[CCCMeasurement.source_value_counts])
                        source_values.append(
                            item[CCCMeasurement.source_values])
                    except KeyError as e:
                        raise type(e), type(e)(
                            str(e.message) +
                            ' not in img {0} plate {1}, pos {2}, {3} '.format(
                                id_image, id_plate, id_row, id_col) +
                            '\nContents:{0}'.format(item)
                        )

    target_value = np.array(ccc[CellCountCalibration.independent_data]).ravel()

    return CalibrationEntry(
        target_value=target_value[inclusion_filter].tolist(),
        source_values=np.array(source_values)[inclusion_filter].tolist(),
        source_value_counts=np.array(
            source_value_counts)[inclusion_filter].tolist(),
        image=None,
        colony_name=None)


def get_calibration_optimization_function(degree=5, include_intercept=False):

    arr = np.zeros((degree + 1,), np.float)

    def poly(x, c1, cn):
        arr[-2] = c1
        arr[0] = cn
        return tuple(v.sum() for v in np.polyval(arr, x))

    def poly_with_intercept(x, m, c1, cn):
        arr[-1] = m
        arr[-2] = c1
        arr[0] = cn
        return tuple(v.sum() for v in np.polyval(arr, x))

    return poly_with_intercept if include_intercept else poly


def get_calibration_polynomial(coefficients_array):

    return np.poly1d(coefficients_array)


def _get_expanded_data(data_store):

    measures = min(len(getattr(data_store, k)) for k in
                   ('target_value', 'source_values', 'source_value_counts'))

    x = np.empty((measures,), dtype=object)
    y = np.zeros((measures,), dtype=np.float64)
    x_min = None
    x_max = None

    values = data_store.source_values
    counts = data_store.source_value_counts
    targets = data_store.target_value

    for pos in range(measures):

        x[pos] = _expand_compressed_vector(
            values[pos], counts[pos], dtype=np.float64)
        y[pos] = targets[pos]

        if x_min is None or x_min > x[pos].min():

            x_min = x[pos].min()

        if x_max is None or x_max < x[pos].max():

            x_max = x[pos].max()

    return x, y, x_min, x_max


def _expand_compressed_vector(values, counts, dtype):

    return np.hstack((np.repeat(value, count)
                      for value, count in izip(values, counts))).astype(dtype)


def poly_as_text(poly):

    def coeffs():
        for i, coeff in enumerate(poly[::-1]):
            yield "{0:.2E} x^{1}".format(coeff, i)

    return "y = {0}".format(" + ".join(coeffs()))


def calculate_polynomial(data_store, degree=5):

    x, y, _, _ = _get_expanded_data(data_store)
    poly = get_calibration_optimization_function(degree)

    p0 = np.zeros((2,), np.float)
    if degree == 5:
        # This is a known solution for a specific set of Sc data
        # it is hopefully a good startingpoint
        p0[0] = 48.99061427688507
        p0[1] = 3.379796310880545e-05

    (c1, cn), pcov = curve_fit(poly, x, y, p0=p0, bounds=[0, np.inf])

    poly_vals = np.zeros((degree + 1))
    poly_vals[-2] = c1
    poly_vals[0] = cn

    _logger.info(
        "Data produced polynomial {0} with 1 sigma per term (x^1, x^{2}) {1}".format(
            poly_as_text(poly_vals), np.sqrt(np.diag(pcov)), degree)
    )

    return poly_vals


def load_calibrations(file_path=None):

    if file_path is None:
        file_path = Paths().analysis_polynomial

    try:

        with open(file_path, 'r') as fh:

            try:
                data = json.load(fh)
            except ValueError:
                data = {}
                fh.seek(0)
                for i, l in enumerate(fh):
                    try:
                        key, value = eval(l)
                        data[key] = value
                    except (TypeError, ValueError):
                        _logger.info(
                            "Skipping line {0}: '{0}' (can't parse)".format(
                                i, l.strip())
                        )

    except IOError:
        _logger.warning("Could not locate file '{0}'".format(file_path))
        data = {}

    return data


def load_calibration(label="", poly_degree=None, file_path=None):

    data = load_calibrations(file_path)
    if poly_degree is not None:
        label = "{0}_{1}".format(label, poly_degree)

    for k in data:

        if k.startswith(label):

            if poly_degree is None:
                _logger.info(
                    "Using polynomial {0}: {1}".format(
                        k, poly_as_text(data[k]))
                )

            return data[k]


def _safe_copy_file_if_needed(file_path):

    # Make copy of previous state
    if os.path.isfile(file_path):

        local_zone = tz.gettz()
        stamp = datetime.fromtimestamp(time.time(), local_zone).isoformat()

        target = "{0}.{1}.polynomials".format(
            file_path.rstrip("polynomials"), stamp)
        shutil.copy(file_path, target)


def add_calibration(label, poly, file_path=None):

    if file_path is None:
        file_path = Paths().analysis_polynomial

    _safe_copy_file_if_needed(file_path)

    data = load_calibrations(file_path)

    key = "{0}_{1}".format(label, len(poly) - 1)
    if key in data:
        _logger.warning(
            "Replacing previous calibration {0}: {1}".format(key, data[key]))

    data[key] = poly.tolist() if hasattr(poly, 'tolist') else poly

    with open(file_path, 'w') as fh:

        json.dump(data, fh)


def remove_calibration(label, degree=None, file_path=None):

    if file_path is None:
        file_path = Paths().analysis_polynomial

    data = load_calibrations(file_path)
    keys = tuple(data.keys())
    has_changed = False

    for key in keys:

        if degree:

            if key == "{0}_{1}".format(label, degree):
                del data[key]
                has_changed = True
                break
        elif key.startswith("{0}_".format(label)):
            del data[key]
            has_changed = True

    if has_changed:
        _safe_copy_file_if_needed(file_path)
        with open(file_path, 'w') as fh:
            json.dump(data, fh)

        return True

    else:
        _logger.warning(
            "No polynomial was found matching the criteria (label={0}, degree={1}".format(
                label, degree)
        )
        return False


def _get_all_grid_shapes(ccc):

    plates = []
    cells = []

    for image in ccc[CellCountCalibration.images]:
        for key in sorted(image[CCCImage.plates].keys()):
            plate = image[CCCImage.plates][key]
            if plate[CCCPlate.grid_shape]:
                plates.append(plate[CCCPlate.grid_shape])
                cells.append(plate[CCCPlate.compressed_ccc_data])

    return plates, cells


@_validate_ccc_edit_request
def add_external_data_to_ccc(identifier, data_file, report):

    warnings = report.get('warnings', [])
    report['warnings'] = warnings

    errors = report.get('errors', [])
    report['errors'] = errors

    filetype = data_file.filename.split(".")[-1].lower()
    if filetype not in ['.xls', '.xlsx', '.csv']:
        errors.append('File format {0} not supported'.format(filetype))
        return False

    ccc = __CCC[identifier]

    file_path = Paths().ccc_external_data_pattern.format(identifier, filetype)
    data_file.save(file_path)
    ccc[CellCountCalibration.independent_data_source] = filetype

    grid_shapes, measurements = _get_all_grid_shapes(ccc)
    meta_data = MetaData(grid_shapes, file_path)
    if not meta_data.loaded:
        errors.append(
            'File could not be understood in terms of plates included'
            '(Grid shapes: {0})'.format(grid_shapes))
        return False

    data_errors = False
    for id_plate, measurement_set in enumerate(measurements):
        for id_outer, outer in enumerate(measurement_set):
            for id_inner, compressed_measurement in enumerate(outer):
                measured = compressed_measurement[CCCMeasurement.included]
                independent_data = meta_data[id_plate][id_outer][id_inner][-1]
                independent_data = (
                    (isinstance(independent_data, float) or
                     isinstance(independent_data, int)) and
                    independent_data > 0)

                if independent_data and not measured:
                    warnings.append(
                        "Plate {0}, Pos ({1}, {2}) is not included but has independent data {3}".format(
                            id_plate, id_outer, id_inner,
                            meta_data[id_plate][id_outer][id_inner])
                    )

                elif not independent_data and measured:

                    errors.append(
                        "Plate {0}, Pos ({1}, {2}) is included but has no valid independent data {3}".format(
                            id_plate, id_outer, id_inner,
                            meta_data[id_plate][id_outer][id_inner])
                    )

                    data_errors = True

    if data_errors:
        return False

    ccc[CellCountCalibration.independent_data] = (
        meta_data.get_column_index_from_all_plates(-1))

    return save_ccc_to_disk(ccc)


@_validate_ccc_edit_request
def construct_polynomial(identifier, poly_name, power):

    ccc = __CCC[identifier]
    data_store = _collect_all_included_data(ccc)
    poly_coeffs = calculate_polynomial(data_store, power).tolist()
    poly = get_calibration_polynomial(poly_coeffs)

    validation = validate_polynomial(data_store, poly)

    if validation is not CalibrationValidation.OK:
        return {
            'validation': validation
        }
    _add_poly(ccc, poly_name, power, poly_coeffs)
    if not save_ccc_to_disk(identifier):
        return False

    # Darkening -> Cell Count Per pixel
    # Then multiply by number of such pixels in colony
    # Then Sum cell counts per colony
    calc_sizes = [
        (poly(intensities) * counts).sum()
        for intensities, counts in
        zip(
            data_store.source_values,
            data_store.source_value_counts)]

    return {
        'ccc': identifier,
        'polynomial_coefficients': poly_coeffs,
        'polynomial_name': poly_name,
        'polynomial_degree': power,
        'measured_sizes': data_store.target_value,
        'calculated_sizes': calc_sizes,
        'validation': validation,
    }


def _add_poly(ccc, poly_name, power, poly_coeffs):

    ccc[poly_name] = {"power": power, "coefficients": poly_coeffs}
