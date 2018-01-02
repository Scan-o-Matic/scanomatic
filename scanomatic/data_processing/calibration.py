import numpy as np
from enum import Enum
from itertools import izip
from scipy.optimize import leastsq
from scipy.stats import linregress
import re
from uuid import uuid1
from collections import namedtuple


from scanomatic.generics.maths import mid50_mean
from scanomatic.io.logger import Logger
from scanomatic.io.paths import Paths
from scanomatic.io.fixtures import Fixtures
from scanomatic.image_analysis.image_basics import (
    load_image_to_numpy, Image_Transpose)
from scanomatic.image_analysis.first_pass_image import FixtureImage
from scanomatic.io.ccc_data import (
    CCCImage, CCCPlate, CCCPolynomial, CCCMeasurement, CellCountCalibration,
    CalibrationEntryStatus, get_empty_ccc_entry, load_cccs,
    get_polynomal_entry, get_empty_image_entry, save_ccc,
    validate_polynomial_format,
)

__CCC = {}
_logger = Logger("CCC")

CalibrationData = namedtuple(
    "CalibrationData", [
        'source_values',
        'source_value_counts',
        'target_value',
    ]
)


class ActivationError(Exception):
    pass


class CCCConstructionError(Exception):
    pass


class CalibrationValidation(Enum):
    OK = 0
    """:type : CalibrationValidation"""
    BadSlope = 1
    """:type : CalibrationValidation"""
    BadStatistics = 3
    """:type : CalibrationValidation"""
    BadData = 4


def _ccc_edit_validator(identifier, **kwargs):

    if identifier in __CCC:

        ccc = __CCC[identifier]

        if ("access_token" in kwargs and
                ccc[CellCountCalibration.edit_access_token] ==
                kwargs["access_token"] and
                kwargs["access_token"]):

            if (ccc[CellCountCalibration.status] ==
                    CalibrationEntryStatus.UnderConstruction):

                return True

            else:

                _logger.error(
                    "Can not modify {0} since not under construction".format(
                        identifier)
                )

        else:

            _logger.error(
                "Bad access token for {}, request refused using {}"
                .format(
                    identifier,
                    kwargs.get('access_token', '++NO TOKEN SUPPLIED++'),
                )
            )
    else:

        _logger.error("Unknown CCC {0}".format(identifier))
    return False


def _validate_ccc_edit_request(f):

    def wrapped(identifier, *args, **kwargs):

        if _ccc_edit_validator(identifier, **kwargs):
            del kwargs["access_token"]
            return f(identifier, *args, **kwargs)

    return wrapped


@_validate_ccc_edit_request
def is_valid_edit_request(identifier):
    return True


def get_empty_ccc(species, reference):
    ccc_id = _get_ccc_identifier(species, reference)
    if ccc_id is None:
        return None
    return get_empty_ccc_entry(ccc_id, species, reference)


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


def _insert_default_ccc():

    ccc = get_empty_ccc(
        species='S. cerevisiae',
        reference='Zackrisson et. al. 2016',
    )
    ccc[CellCountCalibration.identifier] = 'default'
    ccc[CellCountCalibration.polynomial] = {
        CCCPolynomial.coefficients:
            (3.379796310880545e-05, 0., 0., 0., 48.99061427688507, 0.),
        CCCPolynomial.power: 5,
    }
    ccc[CellCountCalibration.edit_access_token] = None
    ccc[CellCountCalibration.status] = CalibrationEntryStatus.Active

    __CCC[ccc[CellCountCalibration.identifier]] = ccc


def reload_cccs():

    __CCC.clear()
    _insert_default_ccc()

    for ccc in load_cccs():
        if ccc[CellCountCalibration.identifier] in __CCC:

            _logger.error(
                "Duplicated identifier {0} is not allowed!".format(
                    ccc[CellCountCalibration.identifier])
            )

        else:

            __CCC[ccc[CellCountCalibration.identifier]] = ccc


def get_active_cccs():

    return {
        identifier: ccc for identifier, ccc in __CCC.iteritems()
        if ccc[CellCountCalibration.status] == CalibrationEntryStatus.Active}


def get_polynomial_coefficients_from_ccc(identifier):

    ccc = __CCC[identifier]
    if ccc[CellCountCalibration.status] != CalibrationEntryStatus.Active:
        raise KeyError

    return ccc[CellCountCalibration.polynomial][CCCPolynomial.coefficients]


def get_under_construction_cccs():

    return {
        identifier: ccc for identifier, ccc in __CCC.iteritems()
        if ccc[CellCountCalibration.status] ==
        CalibrationEntryStatus.UnderConstruction}


def add_ccc(ccc):

    if (ccc[CellCountCalibration.identifier] and
            ccc[CellCountCalibration.identifier] not in __CCC):

        __CCC[ccc[CellCountCalibration.identifier]] = ccc

        save_ccc_to_disk(ccc)
        return True

    else:

        _logger.error(
            "'{0}' is not a valid new CCC identifier".format(
                ccc[CellCountCalibration.identifier])
        )
        return False


def has_valid_polynomial(ccc):
    poly = ccc[CellCountCalibration.polynomial]
    try:
        validate_polynomial_format(poly)
    except ValueError as err:
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

    return save_ccc_to_disk(ccc)


@_validate_ccc_edit_request
def delete_ccc(identifier):

    ccc = __CCC[identifier]

    ccc[CellCountCalibration.status] = CalibrationEntryStatus.Deleted
    ccc[CellCountCalibration.edit_access_token] = uuid1().hex

    return save_ccc_to_disk(ccc)


def save_ccc_to_disk(ccc):

    if ccc[CellCountCalibration.identifier] in __CCC:

        return save_ccc(ccc)

    else:

        _logger.error("Unknown CCC identifier {0} ({1})".format(
            ccc[CellCountCalibration.identifier], __CCC.keys()))
        return False


@_validate_ccc_edit_request
def add_image_to_ccc(identifier, image):

    ccc = __CCC[identifier]
    im_json = get_empty_image_entry(ccc)
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
            CCCPlate.grid_shape: None,
            CCCPlate.compressed_ccc_data: {}
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
                        identifier, image_identifier, id_plate)))
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
        return False

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
        identifier, image_identifier, plate_id, x, y, cell_count,
        image, blob_filter, background_filter):

    ccc = __CCC[identifier]
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

    plate[CCCPlate.compressed_ccc_data][(x, y)] = {
        CCCMeasurement.source_value_counts: counts,
        CCCMeasurement.source_values: values,
        CCCMeasurement.cell_count: cell_count,
    }

    return save_ccc_to_disk(ccc)


def calculate_sizes(data, poly):
    """Get summed population size using a CCC.

    Use pixel darkening -> Cell Count Per pixel (CCC polynomial)
    Then multiply by number of such pixels in colony
    Then Sum cell counts per colony
    """
    return [
        (poly(values) * counts).sum()
        for values, counts in
        zip(data.source_values, data.source_value_counts)
    ]


def validate_polynomial(slope, p_value, stderr):

    if abs(1.0 - slope) > 0.1:
        _logger.error("Bad slope for polynomial: {0}".format(slope))
        return CalibrationValidation.BadSlope

    if p_value > 0.01 or stderr > 0.05:
        return CalibrationValidation.BadStatistics

    return CalibrationValidation.OK


def _collect_all_included_data(ccc):
    source_values = []
    source_value_counts = []
    target_value = []

    for id_image, image_data in enumerate(ccc[CellCountCalibration.images]):

        for id_plate, plate in image_data[CCCImage.plates].items():

            for colony_data in plate[CCCPlate.compressed_ccc_data].values():

                source_value_counts.append(
                    colony_data[CCCMeasurement.source_value_counts])
                source_values.append(colony_data[CCCMeasurement.source_values])
                target_value.append(colony_data[CCCMeasurement.cell_count])

    sort_order = np.argsort(target_value)
    source_values = [source_values[colony] for colony in sort_order]
    source_value_counts = [
        source_value_counts[colony] for colony in sort_order
    ]
    source_value_sorts = [np.argsort(vector) for vector in source_values]
    return CalibrationData(
        source_values=[
            np.array(vector)[sort].tolist() for vector, sort in
            zip(source_values, source_value_sorts)
        ],
        source_value_counts=[
            np.array(vector)[sort].tolist() for vector, sort in
            zip(source_value_counts, source_value_sorts)
        ],
        target_value=np.array([
            target_value[sort] for sort in sort_order
        ])
    )


def get_calibration_optimization_function(degree=5):

    coeffs = np.zeros((degree + 1,), np.float)

    def poly(data_store, *guess):
        coeffs[:-1] = np.exp(guess)
        return tuple(
            (np.polyval(coeffs, values) * counts).sum()
            for values, counts in
            zip(data_store.source_values, data_store.source_value_counts))

    return poly


def get_calibration_polynomial_residuals(
        guess, colony_sum_function, data_store):

    return data_store.target_value - colony_sum_function(data_store, *guess)


def get_calibration_polynomial(coefficients_array):

    return np.poly1d(coefficients_array)


def poly_as_text(poly):

    def coeffs():
        for i, coeff in enumerate(poly[::-1]):
            if (coeff != 0):
                yield "{0:.2E} x^{1}".format(coeff, i)

    return "y = {0}".format(" + ".join(coeffs()))


def calculate_polynomial(data_store, degree=5):

    fit_function = get_calibration_optimization_function(degree)

    p0 = np.zeros((degree,), np.float)
    if degree == 5:
        # This is a known solution for a specific set of Sc data
        # it is hopefully a good startingpoint
        p0[:] = np.log([
            5.263*10**-5,
            4.012*10**-3,
            3.962*10**-2,
            0.9684,
            2.008*10**-6,
        ])
    try:
        poly_vals, _ = leastsq(
            get_calibration_polynomial_residuals,
            p0,
            args=(fit_function, data_store)
        )
    except TypeError:
        raise CCCConstructionError("Invalid data (probably too little)")

    poly_vals = np.r_[np.exp(poly_vals), 0]

    _logger.info(
        "Produced {} degree polynomial {}".format(
            degree, poly_as_text(poly_vals)
        )
    )
    return poly_vals


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
def construct_polynomial(identifier, power):

    ccc = __CCC[identifier]
    data_store = _collect_all_included_data(ccc)
    try:
        poly_coeffs = calculate_polynomial(data_store, power).tolist()
    except CCCConstructionError:
        return {
            'validation': CalibrationValidation.BadData
        }
    poly = get_calibration_polynomial(poly_coeffs)

    calculated_sizes = calculate_sizes(data_store, poly)
    slope, intercept, _, p_value, stderr = linregress(
        data_store.target_value, calculated_sizes)

    validation = validate_polynomial(slope, p_value, stderr)

    if validation is CalibrationValidation.OK:
        ccc[CellCountCalibration.polynomial] = get_polynomal_entry(
            power, poly_coeffs,
        )

        if not save_ccc_to_disk(ccc):
            return False

    return {
        'polynomial_coefficients': poly_coeffs,
        'measured_sizes': data_store.target_value.tolist(),
        'calculated_sizes': calculated_sizes,
        'validation': validation.name,
        'correlation': {
            'slope': slope,
            'intercept': intercept,
            'p_value': p_value,
            'stderr': stderr,
        },
    }


def get_all_colony_data(identifier):
    ccc = __CCC[identifier]
    data_store = _collect_all_included_data(ccc)
    if data_store.source_values:
        min_values = min(min(vector) for vector in data_store.source_values)
        max_values = max(max(vector) for vector in data_store.source_values)
    else:
        min_values = 0
        max_values = 0
    if data_store.source_value_counts:
        max_counts = max(
            max(vector) for vector in data_store.source_value_counts
        )
    else:
        max_counts = 0
    return {
        'source_values': data_store.source_values,
        'source_value_counts': data_store.source_value_counts,
        'target_values': data_store.target_value.tolist(),
        'min_source_values': min_values,
        'max_source_values': max_values,
        'max_source_counts': max_counts,
    }


if not __CCC:
    reload_cccs()
