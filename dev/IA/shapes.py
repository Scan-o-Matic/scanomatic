import types
import numpy as np
from scipy.ndimage import center_of_mass, binary_dilation
from matplotlib import nxutils as nx


def _get_rotated_point(p, rotation):

    p = np.mat(list(p)).T

    R = np.mat([[np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]])

    p = np.array(R * np.mat(p)).T

    return p


def _get_rotated_verts(verts, rotation, origo=None, after_adjust_origo=True):

    if origo is None:

        origo = verts.mean(0)

    pts = verts - origo

    R = np.mat([[np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]])

    pts = np.array(R * np.mat(pts.T)).T

    if after_adjust_origo:
        pts -= pts.min(0)

    return pts


def _get_list(p):

    if type(p) == types.IntType or type(p) == types.FloatType:

        p = [p, p]

    return p


def _get_n_shape_points(n, radius, initial_rotation=0,
    adjust_for_no_negatives=True):

    if n < 3:

        return None

    d_step = 2 * np.pi / n
    v = np.arange(2) * radius
    verts = list()

    for i in xrange(n):

        p = _get_rotated_point(v, i * d_step + initial_rotation)
        verts.append(p.ravel())

    verts = np.array(verts)

    if adjust_for_no_negatives:
        verts -= verts.min(0)

    return verts


def _points_in_circle(i0, j0, r):

    def intceil(x):
        return int(np.round(x))

    for i in xrange(intceil(i0 - r), intceil(i0 + r)):

        ri = np.sqrt(r ** 2 - (i - i0) ** 2)

        for j in xrange(intceil(j0 - ri), intceil(j0 + ri)):

            yield (i, j)


def place_im_in_im(large_im, small_im, anchor_at, use_mass_center=True):
    """places small_im in large_im at anchor site (upper left corner, or
    mass center)"""

    if use_mass_center:

        mass_center = map(np.round, center_of_mass(small_im))
        anchor_at = [anchor_at[i]-mass_center[i] 
                        for i in xrange(len(anchor_at))]


    l_anchor_at = [(a>=0 and a or 0) for a in anchor_at]
    s_anchor_at = [(a<0 and a*-1 or 0) for a in anchor_at]

    upper_bounds = [(l_anchor_at[i] + (small_im.shape[i] - s_anchor_at[i])
                    > large_im.shape[i] and large_im.shape[i] or 
                    l_anchor_at[i] + (small_im.shape[i] - s_anchor_at[i]))
                    for i in xrange(len(anchor_at))]


    large_im[l_anchor_at[0]: upper_bounds[0], l_anchor_at[1]: upper_bounds[1]]\
        [small_im[s_anchor_at[0]: upper_bounds[0] - anchor_at[0],
        s_anchor_at[1]: upper_bounds[1] - anchor_at[1]]==1] = 1

    return large_im


def get_padded(A, p):
    """Returns a padded version of A with padding size p (or p[0], p[1]).
    """

    p = _get_list(p)

    B_shape = [2*p[i] + A.shape[i] for i in xrange(A.ndim)]
    B = np.zeros(B_shape)

    B[p[0]: p[0] + A.shape[0], p[1]: p[1] + A.shape[1]] = A

    return B


def get_rect(size, padding=None, rotation=None, **kwargs):
    """Returns rect of size, if size is Int, then a square"""

    size = _get_list(size)

    if rotation is not None:

        verts = np.array([[0, 0], [size[0], 0], [size[0], size[1]],
            [0, size[1]]])

        verts = _get_rotated_verts(verts, rotation)

        A = get_polygon(verts, **kwargs)

    else:

        A = np.ones(size)

    if padding is not None:
    
        A = get_padded(A, padding)

    return A


def get_star_polygon(spikes, outer_radius, inner_radius,
        initial_rotation=0, padding=None):
    """Returns a star-shaped polygon"""

    inner_offset =  np.pi / spikes
    
    outer_points = _get_n_shape_points(spikes, outer_radius,
        initial_rotation=initial_rotation,
        adjust_for_no_negatives=False)

    inner_points = _get_n_shape_points(spikes, inner_radius,
        initial_rotation=initial_rotation + inner_offset,
        adjust_for_no_negatives=False)

    origo_shift = outer_points.min(0)

    outer_points -= origo_shift
    inner_points -= origo_shift

    points = list()

    for i in xrange(spikes):

        points.append(outer_points[i])
        points.append(inner_points[i])

    A = get_polygon(points)

    if padding is not None:

        A = get_padded(A, padding)

    return A


def get_n_shaped_polygon(n, radius, initial_rotation=0, padding=None):
    """Returns a polygon with n corners"""

    verts = _get_n_shape_points(n, radius, 
        initial_rotation=initial_rotation)

    A = get_polygon(verts)

    if padding is not None:

        A = get_padded(A, padding)

    return A


def get_circle(radius, padding=None):
    """Get an approximate circle"""

    size = [radius*2-1, radius*2]
    A = np.zeros(size)

    raster = np.fromfunction(lambda i, j: 100 + 10 * i + j,
                    size, dtype=int)

    pts_iterator = _points_in_circle(radius-1, radius, radius)

    A[zip(*list(pts_iterator))] = 1

    if padding is not None:

        A = get_padded(A, padding)

    return A

def get_polygon(verts, padding=None, dilate=False):
    """Get a polygon from a list of verts"""

    verts = np.array(verts)

    shape = verts.max(0)
    
    im = np.zeros(shape)

    pts_list = np.where(im==0)

    pts = np.array(zip(*pts_list))

    im.ravel()[nx.points_inside_poly(pts, verts)] = 1
    
    if padding is not None:

        im = get_padded(im)

    if dilate:

        return binary_dilation(im)

    else:

        return im
