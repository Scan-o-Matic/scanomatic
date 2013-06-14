#!/usr/bin/python
"""The Image Time Patcher.

The purpose of this script is to patch time stamps for when the images
where acquired. Generally the issue appears if images were transfered
to other media/drive than the one they were firstly written to, and when
there's a need to re-make the project.

The module can patch times both in the first pass file and in the final
analysis-xml file.
"""

import os
import re
import functools
from argparse import ArgumentParser


def get_time_generator(interval, order="inc"):
    """A simple generator for time intervals.

    :param interval:
        The time in seconds between
    :param order:
        Default value is 'inc' (incremental).
        Other accepted value is 'dec' (decremental)
    :return:
        Generator
    """

    if order.lower() == 'inc':

        t = 0

    else:

        t = 1366013470  # Spring 2013 in unix time
        interval *= -1  # Will make interval negative

    while True:

        yield t
        t += interval


def get_time_generator_from_file(fpath, order='inc'):
    """A simple iterator for the times in the source file.

    :param fpath:
        The path to the first past file
    :param order:
        Default value is 'inc' (incremental).
        Other accepted value is 'dec' (decremental)
    :return:
        Generator
    """

    fh = open(fpath, 'r')
    d = fh.read()
    fh.close()

    times = map(float, re.findall(r"'Time': ([\d.]*)", d))
    times.sort(reverse=order.lower() == 'dec')

    for t in times:

        yield t


def write_times_to_first_pass_file(fpath, timeGenerator):
    """Replaces times in a file with times from the generator

    :param fpath:
        Path to the first pass file
    :param timeGenerator:
        A time generator
    """
    REPLACEMENT = "'Time': {0}"
    PATTERN = r"'Time': [\d.]*"

    fh = open(fpath, 'r')
    d = fh.read()
    fh.close()

    m = re.search(PATTERN, d)
    lTrunc = 0

    while m is not None:

        lBound, rBound = m.span()
        d = (d[:lBound + lTrunc] + REPLACEMENT.format(timeGenerator.next()) +
             d[rBound + lTrunc:])

        lTrunc += rBound
        m = re.search(PATTERN, d[lTrunc:])

    fh = open(fpath, 'w')
    fh.write(d)
    fh.close()


def write_times_to_xml_file(fpath, timeGenerator):
    """Replaces times in a file with times from the generator

    :param fpath:
        Path to the first pass file
    :param timeGenerator:
        A time generator
    """
    REPLACEMENT = "<t>{0}</t>"
    TIME_PATTERN = r'<t>([\d.]*)</t>'
    SCAN_PATTERN = r'<s i="\d*">.*?</s>'

    fh = open(fpath, 'r')
    d = fh.read()
    fh.close()

    m = re.search(SCAN_PATTERN, d)
    lTrunc = 0

    while m is not None:

        m2 = re.search(TIME_PATTERN, m.group())
        lBound, rBound = m.span()

        if m2 is not None:

            l2Bound, r2Bound = m2.span()

            d = (d[:lBound + lTrunc + l2Bound] +
                 REPLACEMENT.format(timeGenerator.next()) +
                 d[lBound + lTrunc + r2Bound:])

        else:

            timeGenerator.next()

        lTrunc += rBound
        m = re.search(SCAN_PATTERN, d[lTrunc:])

    fh = open(fpath, 'w')
    fh.write(d)
    fh.close()


#
#   COMMAND LINE BEHAVIOUR
#

if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__)

    ingroup = parser.add_argument_group("Input Options")

    ingroup.add_argument("-i", "--interval", dest="interval", type=float,
                         help="""A fixed interval between the image (minutes).
                         This should only be used if no original first pass
                         file is accessible""")

    ingroup.add_argument("-t", "--time-file", dest="source_file", type=str,
                         metavar="PATH",
                         help="""The path to a first pass file with correct
                         times in it. (If not supplied, a time interval is
                         needs be supplied.)""")

    outgroup = parser.add_argument_group("Output Options")

    outgroup.add_argument("-f", "--first-pass-file", dest="target_first",
                          type=str, metavar="PATH",
                          help="""First pass file that should be updated with
                          new times""")

    outgroup.add_argument("-x", "--xml-file", dest="target_xml",
                          type=str, metavar="PATH",
                          help="""XML-file that should be updated
                          with new times""")

    args = parser.parse_args()

    if (args.source_file not in (None, "") and
            os.path.isfile(args.source_file)):

        timesF = functools.partial(get_time_generator_from_file,
                                   args.source_file)

    elif args.interval is not None:

        timesF = functools.partial(get_time_generator, args.interval * 60)

    else:

        parser.error("Could not find '{0}' and no interval submitted".format(
            args.source_file))

    doneSomething = False

    if (args.target_first not in (None, "") and os.path.isfile(
            args.target_first)):

        write_times_to_first_pass_file(args.target_first, timesF(order='inc'))
        doneSomething = True

    if (args.target_xml not in (None, "") and os.path.isfile(
            args.target_xml)):

        write_times_to_xml_file(args.target_xml, timesF(order='dec'))
        doneSomething = True

    if not doneSomething:

        parser.error("Could not find any XML or first pass (or non supplied)")

    else:

        print "Done!"
