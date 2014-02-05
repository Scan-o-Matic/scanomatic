import os
import shutil
import sys

import scanomatic.io.logger as logger

_logger = logger.Logger("Post Install")

homeDir = os.path.expanduser("~")

defaltPermission = 0666
installPath = ".scan-o-matic"
defaultSourceBase = "data"

data_files = [
    ('config', {'calibration.polynomials': False,
                'calibration.data': False,
                'grayscales.cfg': False,
                'scan-o-matic.desktop': True}),
    (os.path.join('config', 'fixtures'), {}),
    ('logs', {}),
    ('locks', {}),
    ('images', {'orientation_marker_150dpi.png': True,
                'martin3.png': True,
                'scan-o-matic.png': True})
]


def InstallDataFiles(targetBase=None, sourceBase=None, installList=None):

    if targetBase is None:
        targetBase = os.path.join(homeDir, installPath)

    if sourceBase is None:
        sourceBase = defaultSourceBase

    if installList is None:
        installList = data_files

    if not os.path.isdir(targetBase):
        os.mkdir(targetBase)
        os.chmod(targetBase, 0777)

    for installType in installList:

        curDir, dirFiles = installType
        sourceDir = os.path.join(sourceBase, curDir)
        targetDir = os.path.join(targetBase, curDir)

        if not os.path.isdir(targetDir):

            os.mkdir(targetDir)
            os.chmod(targetDir, 0777)

        for fileName in dirFiles:

            sourcePath = os.path.join(sourceDir, fileName)
            targetPath = os.path.join(targetDir, fileName)

            if (not os.path.isfile(targetPath) and dirFiles[fileName] is None):
                fh = open(targetPath, 'w')
                fh.cloe()
            elif (not os.path.isfile(targetPath) or dirFiles[fileName]
                    or 'y' in raw_input(
                        "Do you want to overwrite {0} (y/N)".format(
                            targetPath)).lower()):

                _logger.info(
                    "Copying file: {0} => {1}".format(
                        sourcePath, targetPath))

                shutil.copy(sourcePath, targetPath)
                os.chmod(targetPath, defaltPermission)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'install':
        InstallDataFiles()
