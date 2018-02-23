#!/usr/bin/env python
from __future__ import absolute_import

import os
import sys
from subprocess import Popen, PIPE, call
import json
from logging import getLogger

package_dependencies = [
    'chardet',
    'enum34',
    'flask',
    'flask-restful',
    'flask_cors',
    'future',
    'matplotlib',
    'numpy',
    'pandas',
    'pillow',
    'prometheus-client',
    'psutil',
    'pytz',
    'requests',
    'scikit-image',
    'scipy',
    'setproctitle',
    'xlrd',
]

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
    "scanomatic.scanning",
    "scanomatic.util",
    "scanomatic.ui_server"
]

silent_install = any(arg.lower() == '--default' for arg in sys.argv)
if silent_install:
    sys.argv = [arg for arg in sys.argv if arg.lower() != '--default']


if len(sys.argv) > 1:

    _logger = getLogger("setup")
    _logger.info("Preparing setup parameters")

    from setuptools import setup
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
                'util/birds.txt',
                'util/adjectives.txt',
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
        install_requires=package_dependencies
    )

    _logger.info("Scan-o-Matic is setup on system")
    _logger.info("Install Complete")
