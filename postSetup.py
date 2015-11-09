import os
import shutil
import sys
import glob

import scanomatic.io.logger as logger

_logger = logger.Logger("Post Install")

homeDir = os.path.expanduser("~")

defaltPermission = 0644
installPath = ".scan-o-matic"
defaultSourceBase = "data"

data_files = [
    ('config', {'calibration.polynomials': False,
                'calibration.data': False,
                'grayscales.cfg': False,
                'rpc.config': False,
                'scan-o-matic.desktop': True}),
    (os.path.join('config', 'fixtures'), {}),
    ('logs', {}),
    ('locks', {}),
    ('images', None),
    ('ui_server', None)
]


def _clone_all_files_in(path):

    for child in glob.glob(os.path.join(path, "*")):
        local_child = child[len(path) + (not path.endswith(os.sep) and 1 or 0):]
        if os.path.isdir(child):
            for grandchild, _ in _clone_all_files_in(child):
                yield os.path.join(local_child, grandchild), True
        else:
            yield local_child, True


def install_data_files(target_base=None, source_base=None, install_list=None):

    if target_base is None:
        target_base = os.path.join(homeDir, installPath)

    if source_base is None:
        source_base = defaultSourceBase

    if install_list is None:
        install_list = data_files

    if not os.path.isdir(target_base):
        os.mkdir(target_base)
        os.chmod(target_base, 0755)

    for install_instruction in install_list:

        relative_directory, files = install_instruction
        source_directory = os.path.join(source_base, relative_directory)
        target_directory = os.path.join(target_base, relative_directory)

        if not os.path.isdir(target_directory):
            os.makedirs(target_directory, 0755)

        if files is None:
            files = dict(_clone_all_files_in(source_directory))
            print files

        for file_name in files:

            source_path = os.path.join(source_directory, file_name)
            target_path = os.path.join(target_directory, file_name)

            os.path.dirname(target_path)

            if not os.path.isdir(os.path.dirname(target_path)):

                os.makedirs(os.path.dirname(target_path), 0755)

            if not os.path.isfile(target_path) and files[file_name] is None:
                _logger.info("Creating file {0}".format(target_path))
                fh = open(target_path, 'w')
                fh.close()
            elif (not os.path.isfile(target_path) or files[file_name]
                    or 'y' in raw_input(
                        "Do you want to overwrite {0} (y/N)".format(
                            target_path)).lower()):

                _logger.info(
                    "Copying file: {0} => {1}".format(
                        source_path, target_path))

                shutil.copy(source_path, target_path)
                os.chmod(target_path, defaltPermission)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'install':
        install_data_files()
