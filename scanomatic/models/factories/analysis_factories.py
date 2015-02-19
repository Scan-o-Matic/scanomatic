__author__ = 'martin'

import os

from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.models.analysis_model import AnalysisModel, GridModel, XMLModel, MEASURES, COMPARTMENTS


class AnalysisModelFactory(AbstractModelFactory):
    _MODEL = AnalysisModel
    STORE_SECTION_HEAD = ("first_pass_file",)
    _SUB_FACTORIES = {
        GridModel: GridModelFactory,
        XMLModel: XMLModelFactory
    }
    STORE_SECTION_SERIALIZERS = {
        ('first_pass_file',): str,
        ('analysis_config_file',): str,
        ('pinning_matrices',): list,
        ('use_local_fixture',): bool,
        ('stop_at_image',): int,
        ('output_directory',): str,
        ('focus_position',): tuple,
        ('suppress_non_focal',): bool,
        ('animate_focal',): bool,
        ('grid_images',): list,
        ('grid_correction',): list,
        ('grid_model',): GridModelFactory,
        ('xml_model',): XMLModelFactory
    }

    @classmethod
    def _validate_first_pass_file(cls, model):

        return cls._is_file(model.first_pass_file)

    @classmethod
    def _validate_analysis_config_file(cls, model):

        return model.analysis_config_file in (None, "") or AbstractModelFactory._is_file(model.analysis_config_file)

    @classmethod
    def _validate_pinning_matrices(cls, model):

        return AbstractModelFactory._is_pinning_formats(model.pinning_matrices)

    @classmethod
    def _validate_use_local_fixture(cls, model):

        return isinstance(model.use_local_fixture, bool)

    @classmethod
    def _validate_stop_at_image(cls, model):

        return isinstance(model.stop_at_image, int)

    @classmethod
    def _validate_output_directory(cls, model):

        return (model.output_directory is None or isinstance(model.output_directory, str) and
                os.sep not in model.output_directory)

    @classmethod
    def _validate_focus_position(cls, model):

        is_coordinate = (cls._is_tuple_or_list(model.focus_position) and
                         all(isinstance(value, int) for value in model.focus_position) and
                         len(model.focus_position) == 3)

        if is_coordinate and cls._validate_pinning_matrices(model):

            plate_exists = (0 <= model.focus_position[0] < len(model.pinning_matrices) and
                            model.pinning_matrices[model.focus_position[0]] is not None)

            return plate_exists and (0 <= val < dim_max for val, dim_max in
                                     zip(model.focus_position[1:], model.pinning_matrices[model.focus_position[0]]))
        return False

    @classmethod
    def _validate_suppress_non_focal(cls, model):

        return isinstance(model.suppress_non_focal, bool)

    @classmethod
    def _validate_animate_focal(cls, model):

        return isinstance(model.animate_focal, bool)

    @classmethod
    def _validate_grid_images(cls, model):

        return model.grid_images is None or (
            cls._is_tuple_or_list(model.grid_images) and
            all(isinstance(val, int) and 0 <= val for val in model.grid_images))

    @classmethod
    def _validate_grid_correction(cls, model):

        def _valid_correction(value):

            return value is None or (len(value) == 2 and all(isinstance(offset, int) for offset in value))

        try:
            return (all(_valid_correction(plate) for plate in model.grid_correction) and
                    len(model.grid_correction) == len(model.pinning_matrices))
        except (TypeError, IndexError):

            return False

    @classmethod
    def _validate_grid_model(cls, model):

        return cls._validate_submodel(model, "grid_model")

    @classmethod
    def _validate_xml_model(cls, model):

        return cls._validate_submodel(model, "xml_model")


class GridModelFactory(AbstractModelFactory):

    _MODEL = GridModel
    STORE_SECTION_SERIALIZERS = {
        ('use_utso',): bool,
        ("median_coefficient",): float,
        ("manual_threshold",): float
    }

    @classmethod
    def _validate_use_utso(cls, model):

        return isinstance(model.use_utso, bool)

    @classmethod
    def _validate_median_coefficient(cls, model):

        return isinstance(model.median_coefficient, float)

    @classmethod
    def _validate_manual_threshold(cls, model):

        return isinstance(model.manual_threshold, float)


class XMLModelFactory(AbstractModelFactory):

    _MODEL = XMLModel
    STORE_SECTION_SERIALIZERS = {
        ("exclude_compartments",): tuple,
        ("exclude_measures",): tuple,
        ("make_short_tag_version",): bool,
        ("short_tag_measure",): MEASURES
    }

    @classmethod
    def _validate_exclude_compartments(cls, model):

        return (cls._is_tuple_or_list(model.exclude_compartments) and
                all(compartment in COMPARTMENTS for compartment in model.exclude_compartments))

    @classmethod
    def _validate_exclude_measures(cls, model):

        return (cls._is_tuple_or_list(model.exclude_measures) and
                all(measure in MEASURES for measure in model.exclude_measures))

    @classmethod
    def _validate_make_short_tag_version(cls, model):

        return isinstance(model.make_short_tag_version, bool)

    @classmethod
    def _validate_short_tag_measure(cls, model):

        return model.short_tage_measure in MEASURES