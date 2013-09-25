#!/usr/bin/env python
"""Script that upgrades earlier version of xml files to later version as long
as it is possible.
"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#DEPENDENCIES
import re
import numpy as np
from argparse import ArgumentParser
import os
import time

#GLOBALS

_CURRENT_VERSION = float(__version__)
_POS_PATTERN = re.compile(r'<gc x="(\d*)" y="(\d*)">')
_POS_REPLACE1 = '<gc x="'
_POS_REPLACE2 = '" y="'
_POS_REPLACE3 = '">'
_PROJECT_TAG_START = r'<ptag>'
_PROJECT_TAG_INSERT_POS = r'</pref>'
_PROJECT_TAG_REPLACE = r'</pref><ptag></ptag>'
_SCAN_TAG_START = r'<sltag>'
_SCAN_TAG_INSERT_POS = r'</ptag>'
_SCAN_TAG_REPLACE = r'</ptag><sltag></sltag>'
_VERSION = '<ver>([\d.]*)</ver>'
_VERSION_REPLACE = '<ver>{0}</ver>'

#METHODS


def _relativeTime(d):

    REPLACEMENT = "<t>{0}</t>"
    TIME_PATTERN = r'<t>([\d.]*)</t>'
    SCAN_PATTERN = r'<s i="\d*">.*?</s>'

    times = map(float, re.findall(r"'Time': ([\d.]*)", d))
    timeGenerator = (t - min(times) for t in times)

    m = re.search(SCAN_PATTERN, d)
    lTrunc = 0
    tMin = None

    while m is not None:

        m2 = re.search(TIME_PATTERN, m.group())
        lBound, rBound = m.span()

        t = timeGenerator.next()
        if tMin is None or t < tMin:
            tMin = t

        if m2 is not None:

            l2Bound, r2Bound = m2.span()

            d = (d[:lBound + lTrunc + l2Bound] +
                 REPLACEMENT.format(t) +
                 d[lBound + lTrunc + r2Bound:])

        lTrunc += rBound
        m = re.search(SCAN_PATTERN, d[lTrunc:])

    return d


def _changePositions(data, axesOperations):

    sHit = False
    hits = np.array(re.findall(_POS_PATTERN, data), dtype=str)[
        axesOperations]

    print "\t1/3: Got {0} entries".format(len(hits))

    tmpList = [None] * len(hits)

    for i, hit in enumerate(hits):

        sHit = re.search(_POS_PATTERN, data)

        tmpList[i] = (data[:sHit.start()] + _POS_REPLACE1 + hit[0] +
                      _POS_REPLACE2 + hit[1] + _POS_REPLACE3)

        data = data[sHit.end():]

        if i % 25000 == 0:
            print "\t2/3: {0:.1f}% processed".format(100.0 * i / len(hits))

    tmpList.append(data)

    print "\t2/3: Built array of xml-fragments"

    data = "".join(tmpList)
    print "\t3/3: Rejoined entries"
    return data


def _replaceIfNot(data, ifMissingPattern, replaceCurrent, replaceWith):

    if re.search(ifMissingPattern, data) is None:

        data = re.sub(replaceCurrent, replaceWith, data)

    return data


def _getVersion(data):

    s = re.search(_VERSION, data)
    try:
        return float(s.groups()[0])
    except:
        return None


def _setVersion(data, version):

    return re.sub(_VERSION, _VERSION_REPLACE.format(version), data)

#RUN TIME BEHAVIOUR

if __name__ == "__main__":

    parser = ArgumentParser(
        description=__doc__)

    parser.add_argument("-p", "--path", dest="paths", type=str, default=None,
                        metavar="PATH", help="Path to xml-file(s) to upgrade",
                        nargs="*")

    args = parser.parse_args()

    print "\n"

    if len(args.paths) == 1:
        print "Did you know you can supply several xml-files at the same time?"
        print "\n"

    for filePath in args.paths:

        startTime = time.time()
        print "\n### Working on {0}".format(filePath)

        if os.path.isfile(filePath) is False:
            parser.error("Could not find file")

        fh = open(filePath, 'r')
        data = fh.read()
        fh.close()
        currentVersion = -1

        while currentVersion != float(__version__):

            currentVersion = _getVersion(data)
            if currentVersion is None:
                parser.error("Version could not be detected")

            elif currentVersion < 0.998:
                parser.error(
                    "Version of XML predates upgrade script, no upgrade possible")

            elif currentVersion < 0.999:
                #First axis is the short axis, which should be flipped
                print "\n--- Upgrading to 0.999"
                data = _changePositions(data, (slice(None, None, -1),
                                               slice(None, None, 1)))

                data = _replaceIfNot(data, _PROJECT_TAG_START,
                                     _PROJECT_TAG_INSERT_POS,
                                     _PROJECT_TAG_REPLACE)

                data = _replaceIfNot(data, _SCAN_TAG_START,
                                     _SCAN_TAG_INSERT_POS,
                                     _SCAN_TAG_REPLACE)

                data = _setVersion(data, 0.999)

            elif currentVersion < 0.9991:

                data = _relativeTime(data)
                data = _setVersion(data, 0.9991)

            else:

                currentVersion = float(__version__)

        deltaTime = time.time() - startTime
        print ("\n--- File upgraded to version {0} (took {1} min {2:.3f} s), " +
               "no more upgrades known").format(currentVersion,
                                                int(deltaTime / 60),
                                                deltaTime % 60)

        fh = open(filePath, 'w')
        fh.write(data)
        fh.close()
