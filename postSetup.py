import os
import shutil
import sys

import scanomatic.io.logger as logger

_logger = logger.Logger("Post Install")

homeDir = os.path.expanduser("~")

defaltPermission = 0666
installPath = ".scan-o-matic"

data_files = []
"""
    (os.path.join('wikiwars', 'config'),
     [os.path.join('data', 'client.cfg'),
      os.path.join('data', 'server.cfg')]),
    (os.path.join('wikiwars', 'level'),
     [os.path.join('data', 'world.pickle')])]
"""


def InstallDataFiles(base=None, installList=None):

    if base is None:
        base = homeDir

    if installList is None:
        installList = data_files

    if not os.path.isdir(os.path.join(base, installPath)):
        os.mkdir(os.path.join(base, installPath))

    for installType in installList:

        curDir, dirFiles = installType

        if not os.path.isdir(os.path.join(base, curDir)):

            os.mkdir(os.path.join(base, curDir))
            os.chmod(os.path.join(base, curDir), 0777)

        for filePath in dirFiles:

            fName = os.path.basename(filePath)

            _logger.info(
                "Installing custom file: {0} => {1}".format(
                    filePath, os.path.join(base, curDir, fName)))

            shutil.copy(filePath, os.path.join(base, curDir, fName))
            os.chmod(os.path.join(base, curDir, fName), defaltPermission)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'install':
        InstallDataFiles()
