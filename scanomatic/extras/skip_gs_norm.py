#!/usr/bin/env python
"""Replaces actualy GS-index values of a project with externally
measured values.

It can be run with one replacement GS-calibration per file in the project.
It can be run with one specified GS-cailbration for all images.
It can be run using an internal default value series.
"""

#
#       DEPENDENCIES
#

from argparse import ArgumentParser
import os

#
#       GLOBAL CONSTANTS
#

KEY_INDICES = 'grayscale_indices'
KEY_VALUES = 'grayscale_values'
WRITE_LINE = '{0}\n'

#
#       GLOBAL DEFAULTS
#

DEFAULT_GS_INDICES_STRING = \
    "82,78,74,70,66,62,58,54,50,46,42,38,34,30,26,22,18,14,10,6,4,2,0"

DEFAULT_GS_TARGET_STRING = \
    """231.53754940711462,204.62055335968378,190.77075098814228,
174.65217391304347,160.87747035573122,148.50988142292491,135.5494071146245,
122.1501976284585,111.20553359683794,100.78260869565217,90.936758893280626,
80.173913043478265,72.015810276679844,63.938405797101453,55.079710144927539,
48.739130434782609,41.980237154150196,34.067193675889328,28.264822134387352,
23.126811594202898,18.920948616600789,17.07509881422925,14.446640316205533"""

#
#       METHODS
#


def _parse_list(s, dtype=int, sep=','):
    """Parses an inputlist."""

    return map(dtype, s.split(sep))


def replace_calibrations(in_path, out_path, indices, values):
    """Replaces image dictionaries with new calibration.

    :param in_path:
        The path to the source file
    :param out_path:
        The path to the output file (can be same as input).
    :param indices:
        The list of indices to overwrite current lists
    :param values:
        The list of values to overwrite the current lists
    """
    #READING
    fh = open(in_path, 'r')
    data = []
    for line in fh:
        data.append(line)
    fh.close()

    #OUTPUTTING AND REWRITING
    fh = open(out_path, 'w')
    for line in data:
        try:
            d = eval(line.strip())
        except:
            fh.write(line)
        else:
            if KEY_INDICES in d and KEY_VALUES in d:
                d[KEY_INDICES] = indices
                d[KEY_VALUES] = values

            fh.write(WRITE_LINE.format(d))

    fh.close()


if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Replaces actualy GS calibration of a project with
        externally measured values.


        (It can be run with one replacement GS-calibration per image in
        the project.) Note yet implemented.

        It can be run with one specified GS-cailbration for all images.

        It can be run using an internal default value series.""")

    parser.add_argument(
        "-f", "--file", type=str, dest="in_path",
        help="""The path to the project's first pass analysis that should be
        reformatted.""",
        metavar="PATH")

    parser.add_argument(
        "-o", "--output", type=str, dest="out_path",
        help="""The path to where the replacement should be written.\n
        If no output path supplied, input file will be overwritten.""",
        metavar="PATH", default="")

    parser.add_argument(
        "-i", "--indices", type=str, dest="indices",
        help="""A list of the calibration index values as a comma separated
        list (without spaces).\nDefault is:\n{0}""".format(
        DEFAULT_GS_INDICES_STRING),
        default=DEFAULT_GS_INDICES_STRING, metavar="LIST")

    parser.add_argument(
        "-v", "--values", type=str, dest="values",
        help="""A list of the calibration target values as comma separated
        list without spaces. Will be used for all images in project.\n
        Default is:\n{0}""".format(DEFAULT_GS_TARGET_STRING),
        default=DEFAULT_GS_TARGET_STRING, metavar="LIST")

    #Evaluating input

    args = parser.parse_args()

    try:
        indices = _parse_list(args.indices)
    except:
        parser.error("Illegal values in list of indices ({0})".format(
            args.indices))

    try:
        values = _parse_list(args.values, dtype=float)
    except:
        parser.error("Illegal values in list of target values ({0})".format(
            args.values))

    if len(indices) != len(values):

        parser.error(
            "Unequal number of indices ({0}) and target values ({1}).".format(
                len(indices), len(values)))

    if not os.path.isfile(args.in_path):
        parser.error(
            "Could not locate input file '{0}'".format(args.in_path))

    if args.in_path in (None, ""):

        args.in_path = args.out_path

    replace_calibrations(args.in_path, args.out_path, indices, values)
    print "DONE!"
