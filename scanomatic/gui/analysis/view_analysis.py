"""The GTK-GUI view"""
__author__ = "Martin Zackrisson"
__copyright__ = "Swedish copyright laws apply"
__credits__ = ["Martin Zackrisson"]
__license__ = "GPL v3.0"
__version__ = "0.9991"
__maintainer__ = "Martin Zackrisson"
__email__ = "martin.zackrisson@gu.se"
__status__ = "Development"

#
# DEPENDENCIES
#


#
# INTERNAL DEPENDENCIES
#

import scanomatic.gui.generic.view_generic as view_generic

#
# STATIC GLOBALS
#

from scanomatic.gui.generic.view_generic import \
    PADDING_SMALL, PADDING_MEDIUM, PADDING_LARGE, PADDING_NONE

#
# CLASSES
#

from view_analysis_about import \
    Analysis_Stage_About, Analysis_Top_Root

from view_analysis_first_pass import \
    Analysis_Stage_First_Pass, Analysis_Stage_First_Pass_Running, \
    Analysis_First_Pass_Top

from view_analysis_inspect import \
    Analysis_Inspect_Top, Analysis_Inspect_Stage

from view_analysis_project import \
    Analysis_Top_Project, Analysis_Stage_Project_Running, \
    Analysis_Stage_Project

from view_analysis_convert import \
    Analysis_Convert_Top, Analysis_Convert_Stage

from view_analysis_single import \
    Analysis_Top_Done, Analysis_Top_Image_Sectioning, \
    Analysis_Top_Auto_Norm_and_Section, Analysis_Top_Image_Normalisation, \
    Analysis_Top_Image_Selection, Analysis_Stage_Auto_Norm_and_Section, \
    Analysis_Top_Image_Plate, Analysis_Stage_Image_Selection, \
    Analysis_Stage_Image_Norm_Manual, Analysis_Stage_Image_Sectioning, \
    Analysis_Stage_Image_Plate, Analysis_Stage_Log

from view_analysis_extract import \
    Analysis_Extract_Top, Analysis_Extract_Stage


class Analysis(view_generic.Page):

    def __init__(self, controller, model, top=None, stage=None):

        super(Analysis, self).__init__(controller, model, top=top,
                                       stage=stage)

    def _default_top(self):

        widget = Analysis_Top_Root(self._controller, self._model)

        return widget

    def _default_stage(self):

        widget = Analysis_Stage_About(self._controller, self._model)

        return widget

