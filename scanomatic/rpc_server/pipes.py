"""Factory for pipe effectors"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# INTERNAL DEPENDENCIES
#

import subprocs

#
# CLASSES
#


class PipeEffector(object):

    def __new__(cls, pipeType, pipe):

        if cls is PipeEffector:
            if pipeType is subprocs.AnalysisEffector:

