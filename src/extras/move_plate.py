#!/usr/bin/env python
"""The script moves a plate from one index to another (empty) index.

The purpose is if user has malplace the plates in the scanner, violating
the layout in the planner.

Note that plates can only be moved to empty positions, so if there are no
empty, (e.g. 4 out of 4 used.) then move it to a novel position (e.g. 5).

The number of plates will be the highest index when all moves are done.
"""

from argparse import ArgumentParser
import os
import re


def getMovePlate(data, source, target):
    """Moves a plate to new position.

    :param data:
        String-like object with the xml-data
    :param source:
        Source index
    :param target:
        Target index
    :return:
        Updated data (or original data if move conflict)
    """

    SEARCH_PATTERN = r'<p i="{0}">'
    SEARCH_PM = r'<p-m i="{0}">(.*?)</p-m>'
    SEARCH_REPLACE = '<p-m i="{0}">\1</p-m>'

    m = re.search(SEARCH_PATTERN.format(target), data)

    if m is not None:

        print "Error:\tTarget plate already exists, skipping"
        return data

    data = re.sub(SEARCH_PM.format(source), SEARCH_REPLACE.format(target), data)

    return re.sub(SEARCH_PATTERN.format(source),
                  SEARCH_PATTERN.format(target), data)

#
#   COMMAND LINE BEHAVIOUR
#

if __name__ == "__main__":

    P_INDEX = "PLATE INDEX"
    MOVE_HELP = """{0} plate position, numbering starting at 1,
    for the {1} move operation"""

    parser = ArgumentParser(
        description=__doc__)

    parser.add_argument("-p", "--path", dest="path", type=str, default=None,
                        metavar="PATH", help="Path to xml-file to patch")

    parser.add_argument("-s1", dest="s1", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Source', '1st'))

    parser.add_argument("-t1", dest="t1", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Target', '1st'))

    parser.add_argument("-s2", dest="s2", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Source', '2nd'))

    parser.add_argument("-t2", dest="t2", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Target', '2nd'))

    parser.add_argument("-s3", dest="s3", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Source', '3rd'))

    parser.add_argument("-t3", dest="t3", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Target', '3rd'))

    parser.add_argument("-s4", dest="s4", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Source', '4th'))

    parser.add_argument("-t4", dest="t4", type=int, default=None,
                        metavar=P_INDEX,
                        help=MOVE_HELP.format('Target', '4th'))

    args = parser.parse_args()

    if args.path is None or os.path.isfile(args.path) is False:

        parser.error("Could not localize file '{0}'".format(args.path))

    print "*Info:\tReading Data..."

    fh = open(args.path, 'r')
    data = fh.read()
    fh.close()

    for i in range(1, 5):

        source = getattr(args, "s{0}".format(i))
        target = getattr(args, "t{0}".format(i))

        if source is not None and target is not None:

            print "*Info:\tMoving step {0} {1} -> {2}".format(i, source,
                                                              target)

            data = getMovePlate(data, source - 1, target - 1)

        elif source is not None or target is not None:

            print """*Error:\tIncomplete move instructions {0} -> {1}
            for move {2}""".format(source, target, i)

    print "*Info:\tSaving Results!"

    fh = open(args.path, 'w')
    fh.write(data)
    fh.close()

    print "*Info:\tDone!"
