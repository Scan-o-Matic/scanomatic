#!/usr/bin/env python


#
# DEPENDENCIES
#

import os
import sys
from subprocess import Popen, PIPE, call
import json


#
# PREPARING INSTALLATION
#

package_dependencies = [
    'argparse', 'matplotlib', 'multiprocessing', 'odfpy',
    'numpy', 'sh', 'nmap', 'configparse', 'skimage',
    'uuid', 'PIL', 'scipy', 'setproctitle', 'psutil', 'flask', 'requests', 'pandas']

scripts = [
    os.path.join("scripts", p) for p in [
        "scan-o-matic",
        "scan-o-matic_server",
    ]
]

packages = [
    "scanomatic",
    "scanomatic.generics",
    "scanomatic.models",
    "scanomatic.models.factories",
    "scanomatic.io",
    "scanomatic.qc",
    "scanomatic.server",
    "scanomatic.image_analysis",
    "scanomatic.data_processing",
    "scanomatic.data_processing.phases",
    "scanomatic.util",
    "scanomatic.ui_server"
]

#
# Parsing and removing argument for accepting all questions as default
#

silent_install = any(arg.lower() == '--default' for arg in sys.argv)
if silent_install:
    sys.argv = [arg for arg in sys.argv if arg.lower() != '--default']

#
# Parsing and removing arguments for branch information
#

branch = None
branch_info = tuple(i for i, arg in enumerate(sys.argv) if arg.lower() == '--branch')

if branch_info:
    branch_info = branch_info[0]
    branch = sys.argv[branch_info + 1] if len(sys.argv) > branch_info + 1 else None
    sys.argv = sys.argv[:branch_info] + sys.argv[branch_info + 2:]


#
# Parsing and removing version upgrade in argument
#

version_update = {i: v for i, v in enumerate(sys.argv) if v.lower().startswith("--version")}
if version_update:
    id_argument = version_update.keys()[0]
    sys.argv = sys.argv[:id_argument] + sys.argv[id_argument + 1:]
    version_update = version_update[id_argument].lower().split("-")[-2:]
    version_update[0] = True
    version_update[1] = version_update[1] if version_update[1] in ('minor', 'major') else False


#
# Python-setup
#

from setup_tools import MiniLogger, patch_bashrc_if_not_reachable, test_executable_is_reachable

if len(sys.argv) > 1:

    if sys.argv[1] == 'uninstall':
        call('python -c"from setup_tools import uninstall;uninstall()"', shell=True)
        sys.exit()

    _logger = MiniLogger()
    _logger.info("Checking non-python dependencies")

    #
    # INSTALLING NON-PYTHONIC PROGRAMS
    #

    program_dependencies = ('nmap',)
    PROGRAM_NOT_FOUND = 32512
    install_dependencies = []

    for dep in program_dependencies:
        try:
            p = Popen(dep, stdout=PIPE, stderr=PIPE)
            p.communicate()
        except OSError:
            install_dependencies.append(dep)

    if len(install_dependencies) > 0:

        if os.name == 'posix':

            if os.system("gksu apt-get install {0}".format(
                    " ".join(install_dependencies))) != 0:

                _logger.warning("Could not install: {0}".format(
                    install_dependencies))

        else:

            _logger.warning(
                "Scan-o-Matic is only designed to be run on Linux. "
                "Setup will try to continue but you are on your own from now on. "
                "The following programs were not found: {0}".format(
                    install_dependencies))


    _logger.info("Non python dependencies done")
    _logger.info("Preparing setup parameters")
    from setup_tools import update_init_file

    if version_update:

        #
        # PRE-INSTALL VERSIONING
        #

        from setup_tools import get_hash_all_files, get_package_hash, get_hash

        _logger.info("Checking for local changes")

        hasher = get_package_hash(packages)
        get_hash(scripts, hasher=hasher)
        get_hash(["setup.py", "setup_tools.py"], hasher=hasher)
        get_hash_all_files("data", depth=5, hasher=hasher)
        cur_hash = hasher.hexdigest()

        try:
            with open("version.hash", 'rb') as fh:
                prev_hash = fh.read()
        except IOError:
            prev_hash = None

        if prev_hash != cur_hash or version_update[1]:

            _logger.info("Local changes detected")

            update_init_file(release=version_update[1])

            _logger.info("Updated version")

            hasher = get_package_hash(packages)
            get_hash(scripts, hasher=hasher)
            get_hash(["setup.py", "setup_tools.py"], hasher=hasher)
            get_hash_all_files("data", depth=5, hasher=hasher)
            cur_hash = hasher.hexdigest()

            with open("version.hash", 'wb') as fh:
                fh.write(cur_hash)

        else:

            _logger.info("No local changes detected!")
    else:
        update_init_file(do_branch=True, do_version=False)
        _logger.info("Skipping checking changes to current version")
    #
    # INSTALLING SCAN-O-MATIC
    #

    from distutils.core import setup
    from scanomatic.__init__ import get_version
    _logger.info("Setting up Scan-o-Matic on the system")

    setup(
        name="Scan-o-Matic",
        version=get_version(),
        description="High Throughput Solid Media Image Phenotyping Platform",
        long_description="""Scan-o-Matic is a high precision phenotyping platform
        that uses scanners to obtain images of yeast colonies growing on solid
        substrate.

        The package contains a user interface as well as an extensive package
        for yeast colony analysis from scanned images.
        """,
        author="Martin Zackrisson",
        author_email="martin.zackrisson@gu.se",
        url="www.gitorious.org/scannomatic",
        packages=packages,

        package_data={
            "scanomatic": [
                'ui_server_data/*.html',
                'ui_server_data/js/*.js',
                'ui_server_data/js/external/*.js',
                'ui_server_data/style/*.css',
                'ui_server_data/fonts/*',
                'ui_server_data/templates/*',
                'images/*',
            ]
        },

        scripts=scripts,
        classifiers=[
            'Development Status :: 4 - Beta',
            'Environment :: X11 Application :: GTK',
            'Environment :: Console',
            'Intended Autdience :: Science/Research',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Natural Language :: English',
            'Operating System :: POSIX :: Linux',
            'Programming Language :: Python :: 2.7',
            'Topic :: Scientific/Engineering :: Bio-Informatics'
        ],
        requires=package_dependencies
    )

    if set(v.lower() for v in sys.argv).intersection(('--help', '--help-commands')):

        print """
        SCAN-O-MATIC Specific Setup
        ---------------------------

        setup.py install [options]

        --version   Checks for changes in the code and upgrades version
                    if detected.

        --version-minor
                    Increment version to next minor (e.g. 1.4.43 -> 1.5)

        --version-major
                    Increment verion to next major (e.g. 1.4.43 -> 2.0)

        --default   Will select default option to setup questions.


        setup.py uninstall

            Uninstalls Scan-o-Matic

        """
        sys.exit()

    if os.name == "nt":

        _logger.info(
            "The files in the script folder can be copied to wherever you"
            " want to try to run Scan-o-Matic. Good luck!")

    _logger.info("Scan-o-Matic is setup on system")

    #
    # POST-INSTALL
    #

    from setup_tools import install_data_files

    _logger.info("Copying data and configuration files")
    install_data_files(silent=True)
    patch_bashrc_if_not_reachable(silent=silent_install)
    _logger.info("Post Setup Complete")

    if not test_executable_is_reachable():
        print """

        INFORMATION ABOUT LOCAL / USER INSTALL
        --------------------------------------

        The scripts for launching the Scan-o-Matic
        programs should be directly accessible from
        the commandline by e.g. writing:

            scan-o-matic

        If this is not the case you will have to
        modify your PATH-variable in bash as follows:

            export PATH=$PATH:$HOME/.local/bin/

        If above gives you direct access to Scan-o-Matic
        then you should put that line at the end of your
        .bashrc file, usually located in your $HOME.

        If it doesn't work, you need to check the setup
        output above to see where the files were copied and
        extend the path accordingly.

        Alternatively, if you install Scan-o-Matic for all
        users then the launch scripts should be copied
        into a folder that is already in path.

        If you use a USB-connected PowerManager, make sure
        sispmctl is installed.

    """

    from scanomatic.io.paths import Paths

    try:
        with open(Paths().source_location_file, mode='w') as fh:
            directory = os.path.dirname(os.path.join(os.path.abspath(os.path.expanduser(os.path.curdir)), sys.argv[0]))
            json.dump({'location': directory, 'branch': branch}, fh)

    except IOError:
        _logger.warning("Could not write info for future upgrades. You should stick to manual upgrades")

    # postSetup.CheckDependencies(package_dependencies)

    _logger.info("Install Complete")

    from subprocess import call

    call(["python", "get_installed_version.py"])
