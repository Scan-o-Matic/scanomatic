from scipy.signal import gaussian
import numpy as np
from enum import Enum
import itertools


class PositionOffset(Enum):

    UpperLeft = (0, 0)
    UpperRight = (0, 1)
    LowerLeft = (1, 0)
    LowerRight = (1, 1)


def get_distance_matrix(measure=gaussian(7, 1)):
    """

    :param measure: The per position distance on the original plate
    :type measure: numpy.ndarray

    :return: numpy.ndarry
    """
    distances = np.outer(measure, measure)

    assert distances.ndim == 2 and distances.shape[0] == distances.shape[1] and distances.shape[0] % 2 == 1

    return distances


def get_reference_position_selector(offset=PositionOffset.LowerRight):

    selector = np.zeros((2, 2), dtype=np.bool)
    selector[offset.value] = True
    return selector


def get_reference_position_filter(plate_shape, position_selector=get_reference_position_selector()):

    return np.tile(
        position_selector,
        tuple(plate_shape[i] / position_selector.shape[i] for i in range(len(plate_shape))))


def get_best_reference_for_experiments(
        plate,
        reference_position_filter=None,
        distance_matrix=get_distance_matrix(),
        scale_references=False):

    def most_stable(curves, reference_weights, reference_filter):

        positions = np.array(zip(*np.where(reference_filter)))
        sums = np.zeros((positions.shape[0],))
        for i, pos in enumerate(positions):

            pos = tuple(pos)
            for other in positions[(positions != pos).any(axis=1)]:
                try:
                    sums[i] += np.abs(curves[pos] - curves[tuple(other)]).sum() * reference_weights[pos]
                except IndexError, e:
                    print "---"
                    print "For position", (x, y), "on plate shape", curves.shape
                    print "Comparing", pos, "and", tuple(other)
                    print "Curves have shape", curves.shape
                    print "weights have shape", reference_weights.shape
                    print "Filter has shape", reference_filter.shape
                    print "---"
                    raise e

        sums = np.ma.masked_invalid(sums)
        if sums.any():
            if scale_references:
                return curves[tuple(positions[sums.argmin()])] * scale_vector
            else:
                return curves[tuple(positions[sums.argmin()])]
        return np.zeros((curves.shape[-1], )) * np.nan

    if reference_position_filter is None:
        reference_position_filter = get_reference_position_filter(plate.shape[:2])

    distance_matrix_size = distance_matrix.shape[0]
    offset = int(np.floor(distance_matrix_size / 2))
    reference_plate = np.zeros_like(plate) * np.nan
    dim_x, dim_y = plate.shape[:2]

    if scale_references:
        ravel_references = np.ma.masked_invalid(
            plate[reference_position_filter].reshape(reference_position_filter.sum(), plate.shape[-1]))
        ravel_experiments = np.ma.masked_invalid(
            plate[reference_position_filter == False].reshape(
                reference_position_filter.size - reference_position_filter.sum(), plate.shape[-1]))
        scale_vector = ravel_experiments.mean(axis=0) / ravel_references.mean(axis=0)

    for x, y in itertools.product(range(dim_x), range(dim_y)):

        if reference_position_filter[x, y]:
            continue

        lower_x = max(x - offset, 0)
        upper_x = min(x + offset + 1, dim_x)
        lower_y = max(y - offset, 0)
        upper_y = min(y + offset + 1, dim_y)

        local_reference_filter = reference_position_filter[lower_x: upper_x, lower_y: upper_y]
        local_reference_curves = plate[lower_x: upper_x, lower_y: upper_y]
        local_weights = distance_matrix[
                        0 if x > offset else offset - x: min(distance_matrix_size, dim_x - (x - offset)),
                        0 if y > offset else offset - y: min(distance_matrix_size, dim_y - (y - offset))]

        reference_plate[x, y] = most_stable(local_reference_curves, local_weights, local_reference_filter)

    return reference_plate


def remove_positions_by_offset_and_flatten(plate, offset=PositionOffset.LowerRight):

    out = np.zeros((plate.shape[0] * plate.shape[1] * 3 / 4, plate.shape[2]), dtype=plate.dtype)
    if out.dtype == np.float:
        out *= np.nan

    i = 0
    offset_x, offset_y = offset.value

    for idx, row in enumerate(plate):

        for idy, vector in enumerate(row):

            if not (idx % 2 == offset_x and idy % 2 == offset_y):
                out[i] = vector
                i += 1

    return out


def vector_norm(plate, scale_references=False):

    reference = get_best_reference_for_experiments(plate, scale_references=scale_references)
    return plate - reference