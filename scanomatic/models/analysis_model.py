__author__ = 'martin'

from enum import Enum

import scanomatic.generics.model as model


COMPARTMENTS = Enum("COMPARTMENTS", names=("Total", "Background", "Blob"))

MEASURES = Enum("MEASURES", names=("Count", "Sum", "Mean", "Median", "Perimeter", "IQR", "IQR_Mean", "Centroid"))


class AnalysisModel(model.Model):

    def __init__(self, first_pass_file="", analysis_config_file="", pinning_matrices=tuple(), use_local_fixture=False,
                 stop_at_image=-1, output_directory="", focus_position=None, suppress_non_focal=False,
                 animate_focal=False, grid_images=None,
                 grid_correction=None, grid_model=None, xml_model=None):

        if grid_model is None:
            grid_model = GridModel()

        if xml_model is None:
            xml_model = XMLModel()

        super(AnalysisModel, self).__init__(first_pass_file=first_pass_file,
                                            analysis_config_file=analysis_config_file,
                                            pinning_matrices=pinning_matrices,
                                            use_local_fixture=use_local_fixture,
                                            stop_at_image=stop_at_image,
                                            output_directory=output_directory,
                                            focus_position=focus_position,
                                            suppress_non_focal=suppress_non_focal,
                                            animate_focal=animate_focal,
                                            grid_images=grid_images,
                                            grid_correction=grid_correction,
                                            grid_model=grid_model,
                                            xml_model=xml_model)


class GridModel(model.Model):

    def __init__(self, use_utso=True, median_coefficient=0.99, manual_threshold=0.05):

        super(GridModel, self).__init__(use_utso=use_utso, median_coefficient=median_coefficient,
                                        manual_threshold=manual_threshold)


class XMLModel(model.Model):

    def __init__(self, exclude_compartments=tuple(), exclude_measures=tuple(), make_short_tag_version=True,
                 short_tag_measure=MEASURES.Sum):

        super(XMLModel, self).__init__(exclude_compartments=exclude_compartments,
                                       exclude_measures=exclude_measures,
                                       make_short_tag_version=make_short_tag_version,
                                       short_tag_measure=short_tag_measure)