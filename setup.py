#!/usr/bin/env python
__version__ = "0.9991"


#
# INSTALLING NON-PYTHONIC PROGRAMS
#

import os
from subprocess import Popen, PIPE

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

        if os.system("gksu apt-get install {0} -y".format(
                " ".join(install_dependencies))) != 0:

            print "\n\n*** UNABLE TO INSTALL PROGRAMS:"
            print "\t{0}\n\n".format(install_dependencies)

    else:

        print "\n\n*** SCAN-O-MATIC should be run on debian-based linux"
        print "You need to install the following yourself:"
        print "\t{0}\n\n".format(install_dependencies)

#
# INSTALLING PYTHON DEPENDENCIES
#

from pkg_resources import WorkingSet , DistributionNotFound

#Obtain a list of current packages
working_set = WorkingSet()

package_dependencies = ('cython', 'argparse', 'matplotlib', 'multiprocessing',
    'numpy', 'sh', 'nmap', 'configparse',
    'uuid', 'PIL', 'scipy', 'scikits-image', 'logging', 'unittest', 'pygtk') 

install_dependencies = []

for dep in package_dependencies:

    try:
        working_set.require(dep)
    except DistributionNotFound:
        install_dependencies.append(dep)

if len(install_dependencies) > 0:

    print "\n\n*** THE FOLLOWING DEPENDENCIES WILL BE INSTALLED:"
    print "\t{0}\n\n".format(install_dependencies)

    from setuptools.command.easy_install import main as install

    install(install_dependencies)

#
# INSTALLING SCAN-O-MATIC
#

"""
from distutils.core import setup

setup(name="Scan-o-Matic",
        version=__version__,
        description="High Throughput Solid Media Phenotyping Platform",
        author="Martin Zackrisson",
        author_email="martin.zackrisson@gu.se",
        url="www.gitorious.org/scannomatic",
        )

        #packages=['numpy', 'scipy', 'matplotlib', 'gtk+', 'nmap', 'sh', 'scikits-image',
            ])
"""
