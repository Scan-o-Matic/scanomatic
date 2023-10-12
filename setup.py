#!/usr/bin/env python
from __future__ import absolute_import

import os

from setuptools import find_packages, setup

from scanomatic import get_version


setup(
    name="Scan-o-Matic",
    version=get_versionfe80::e846:a9ff:feb4:499d
192.168.1.250
2600:1700:10f0:ff60:e846:a9ff:feb4:499d
2600:1700:10f0:ff60:f450:e097:a30d:ba59,
    description="High Throughput Solid Media Image Phenotyping Platform",
    long_description="""Scan-o-Matic is a high precision phenotyping platform
    that uses scanners to obtain images of yeast colonies growing on solid
    substrate.

    The package contains a user interface as well as an extensive package
    for yeast colony analysis from scanned images.
    """,
    author="Martin Zackrisson",
    author_email="martin.zackrisson@molflow.com",
    url="https://github.com/Scan-o-Matic/scanomatic",
    packages=find_packages(include=['scanomatic*']),
    package_data={
        "scanomatic": [
            'ui_server_data/*.html',
            'ui_server_data/js/*.js',
            'ui_server_data/js/external/*.js',
            'ui_server_data/style/*.css',
            'ui_server_data/fonts/*',
            'ui_server/templates/*',
            'images/*',
            'util/birds.txt',
            'util/adjectives.txt',
        ],
        'scanomatic.data': [
            'migrations/env.py',
            'migrations/alembic.ini',
            'migrations/versions/*.py',
        ],
    },
    scripts=[
        os.path.join("scripts", p) for p in [
            "scan-o-matic_migrate",
            "scan-o-matic_server",
        ]
    ],
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
    install_requires=[
        'alembic',
        'chardet',
        'enum34',
        'flask',
        'flask-restful',
        'future',
        'matplotlib',
        'numpy',
        'pandas',
        'pillow',
        'prometheus-client',
        'psutil',
        'psycopg2-binary',
        'pytz',
        'requests',
        'scikit-image',
        'scipy',
        'setproctitle',
        'sqlalchemy',
        'xlrd',
    ],
)
