#!/usr/bin/env python
__version__ = "0.9991"

#
# DEPENDENCIES
#

import os
import sys
from subprocess import Popen, PIPE

#
# INTERNAL DEPENDENCIES
#

import scanomatic.io.logger as logger
import postSetup

_logger = logger.Logger("Scan-o-Matic Setup")
_logger.level = _logger.INFO
_logger.info("Checking non-python dependencies")

#
# INSTALLING NON-PYTHONIC PROGRAMS
#

program_dependencies = ('nmap', 'sispmctl')
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

#
# PREPARING INSTALLATION
#

package_dependencies = [
    'argparse', 'matplotlib', 'multiprocessing', 'odf',
    'numpy', 'sh', 'nmap', 'configparse', 'scikits-image',
    'uuid', 'PIL', 'scipy',  'unittest', 'pygtk']

data_files = []

scripts = [
    os.path.join("scripts", p) for p in [
        "scan-o-matic",
        "scan-o-matic_server",
        "scan-o-matic_qc",
        "scan-o-matic_calibration",
        "scan-o-matic_analysis",
        "scan-o-matic_experiment",
        "scan-o-matic_analysis_move_plate",
        "scan-o-matic_analysis_patch_times",
        "scan-o-matic_make_project",
        "scan-o-matic_analysis_skip_gs_norm",
        "scan-o-matic_relauncher",
        "scan-o-matic_analysis_xml_upgrade",
        "scan-o-matic_xml2image_data",
    ]
]

#
# INSTALLING SCAN-O-MATIC
#

from distutils.core import setup

_logger.info("Setting up Scan-o-Matic on the system")

setup(
    name="Scan-o-Matic",
    version=__version__,
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
    install_requires=package_dependencies,
    packages=[
        "scanomatic",
        "scanomatic.io",
        "scanomatic.io.xml",
        "scanomatic.rpc_server",
        "scanomatic.imageAnalysis", "scanomatic.dataProcessing",
        "scanomatic.dataProcessing.visualization",
        "scanomatic.gui",
        "scanomatic.gui.qc",
        "scanomatic.gui.analysis",
        "scanomatic.gui.config",
        "scanomatic.gui.calibration",
        "scanomatic.gui.generic",
        "scanomatic.gui.experiment",
        "scanomatic.gui.server"],

    package_data={"scanomatic": data_files},
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
    ]
)

    #packages=['numpy', 'scipy', 'matplotlib', 'gtk+', 'nmap', 'sh', 'scikits-image',
    #    ])

if (os.name == "nt"):

    _logger.info(
        "The files in the script folder can be copied to wherever you"
        " want to try to run Scan-o-Matic. Good luck!")

_logger.info("Scan-o-Matic is setup on system")

#
# POST-INSTALL
#

_logger.info("Copying data and configuration files")
postSetup.InstallDataFiles()
_logger.info("Post Setup Complete")

if (sys.argv[-1].upper() == "--USER"):
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

"""
_logger.info("Install Complete")
