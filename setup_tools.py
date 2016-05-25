import os
import shutil
import sys
import glob
import stat


class MiniLogger(object):

    def info(self, txt):
        print("INFO: " + txt)

    def warning(self, txt):
        print("WARNING: " + txt)

    def error(self, txt):
        print("ERROR: " + txt)

_logger = MiniLogger()

home_dir = os.path.expanduser("~")

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


_launcher_text = """[Desktop Entry]
Type=Application
Terminal=false
Icon={user_home}/.scan-o-matic/images/scan-o-matic_icon_256_256.png
Name=Scan-o-Matic
Comment=Large-scale high-quality phenomics platform
Exec={executable_path}
Categories=Science;
"""


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
        target_base = os.path.join(home_dir, installPath)

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


def linux_launcher_install():

    user_home = os.path.expanduser("~")
    exec_path = os.path.join(user_home, '.local', 'bin', 'scan-o-matic')
    if not os.path.isfile(exec_path):
        exec_path = os.path.join(os.sep, 'usr', 'local', 'bin', 'scan-o-matic')
    text = _launcher_text.format(user_home=user_home, executable_path=exec_path)
    target = os.path.join(user_home, '.local', 'share', 'applications', 'scan-o-matic.desktop')

    try:
        with open(target, 'w') as fh:
            fh.write(text)
    except IOError:
        _logger.error("Could not install desktop launcher automatically, you have an odd linux system.")
        _logger.info("""You may want to make a manual 'scan-o-matic.desktop' launcher and place it somewhere nice.
        If so, this is what should be its contents:\n\n{0}\n""".format(text))
    else:
        os.chmod(target, os.stat(target)[stat.ST_MODE] | stat.S_IXUSR)
    _logger.info("Installed desktop launcher for linux menu/dash etc.")


def install_launcher():

    if sys.platform.startswith('linux'):
        linux_launcher_install()
    else:
        _logger.warning("Don't know how to install launchers for this os...")


def uninstall():
    _logger.info("Uninstalling")
    uninstall_lib(_logger)
    uninstall_executables(_logger)
    uninstall_launcher(_logger)


def uninstall_lib(l):
    current_location = os.path.abspath(os.curdir)
    os.chdir(os.pardir)
    import shutil

    try:
        import scanomatic as som
        l.info("Found installation at {0}".format(som.__file__))
        if os.path.abspath(som.__file__) != som.__file__ or current_location in som.__file__:
            l.error("Trying to uninstall the local folder, just remove it instead if this was intended")
        else:

            try:
                shutil.rmtree(os.path.dirname(som.__file__))
            except OSError:
                l.error("Not enough permissions to remove {0}".format(os.path.dirname(som.__file__)))

            parent_dir = os.path.dirname(os.path.dirname(som.__file__))
            for egg in glob.glob(os.path.join(parent_dir, "Scan_o_Matic*.egg-info")):
                try:
                    os.remove(egg)
                except OSError:
                    l.error("Not enough permissions to remove {0}".format(egg))

            l.info("Removed installation at {0}".format(som.__file__))
    except (ImportError, OSError):
        l.info("All install location removed")

    l.info("Uninstall complete")
    os.chdir(current_location)


def uninstall_executables(l):

    for path in os.environ['PATH'].split(":"):
        for file_path in glob.glob(os.path.join(path, "scan-o-matic*")):
            l.info("Removing {0}".format(file_path))
            try:
                os.remove(file_path)
            except OSError:
                l.warning("Not enough permission to remove {0}".format(file_path))


def uninstall_launcher(l):

    user_home = os.path.expanduser("~")
    if sys.platform.startswith('linux'):
        target = os.path.join(user_home, '.local', 'share', 'applications', 'scan-o-matic.desktop')
        l.info("Removing desktop-launcher/menu integration at {0}".format(target))
        try:
            os.remove(target)
        except OSError:
            l.info("No desktop-launcher/menu integration was found or no permission to remove it")

    else:
        l.info("Not on linux, no launcher should have been installed.")


def purge():

    uninstall()

    import shutil
    settings = os.path.join(home_dir, ".scan-o-matic")

    try:
        shutil.rmtree(os.path.join(home_dir, settings))
        _logger.info("Setting have been purged")
    except IOError:
        _logger.info("No settings found")


if __name__ == "__main__":

    if len(sys.argv) > 1:
        action = sys.argv[1].lower()

        if action == 'install-settings':
            install_data_files()
        elif action == 'uninstall':
            uninstall()
        elif action == 'purge':
            purge()
        elif action == 'install-launcher':
            install_launcher()
    else:
        _logger.info("Valid options are 'install-settings', 'install-launcher', 'uninstall', 'purge'")