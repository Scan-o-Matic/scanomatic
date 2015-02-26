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
    def set_absolute_paths(cls, model):

        base_path = os.path.dirname(model.first_pass_file)
        model.analysis_config_file = cls._get_absolute_path(model.analysis_config_file, base_path)
        model.output_directory = cls._get_absolute_path(model.output_directory, base_path)

    @classmethod
    def _get_absolute_path(cls, model, path, base_path):

        if os.path.abspath(path) != path:
            return os.path.join(base_path, path)
        return path

    @classmethod
    def _validate_first_pass_file(cls, model):

        if cls._is_file(model.first_pass_file) and os.path.abspath(model.first_pass_file) == model.first_pass_file:
            return True
        return model.FIELD_TYPES.first_pass_file

    @classmethod
    def _validate_analysis_config_file(cls, model):

        if model.analysis_config_file in (None, "") or AbstractModelFactory._is_file(model.analysis_config_file):
            return True
        return model.FIELD_TYPES.analysis_config_file

    @classmethod
    def _validate_pinning_matrices(cls, model):

        if AbstractModelFactory._is_pinning_formats(model.pinning_matrices):
            return True
        return model.FIELD_TYPES.pinning_matrices

    @classmethod
    def _validate_use_local_fixture(cls, model):

        if isinstance(model.use_local_fixture, bool):
            return True
        return model.FIELD_TYPES.use_local_fixture

    @classmethod
    def _validate_stop_at_image(cls, model):

        if isinstance(model.stop_at_image, int):
            return True
        return model.FIELD_TYPES.stop_at_image

    @classmethod
    def _validate_output_directory(cls, model):

        if (model.output_directory is None or isinstance(model.output_directory, str) and
                os.sep not in model.output_directory):
            return True
        return model.FIELD_TYPES.output_directory

    @classmethod
    def _validate_focus_position(cls, model):

        if model.focus_position is None:
            return True

        is_coordinate = (cls._is_tuple_or_list(model.focus_position) and
                         all(isinstance(value, int) for value in model.focus_position) and
                         len(model.focus_position) == 3)

        if is_coordinate and cls._validate_pinning_matrices(model):

            plate_exists = (0 <= model.focus_position[0] < len(model.pinning_matrices) and
                            model.pinning_matrices[model.focus_position[0]] is not None)

            if plate_exists and (0 <= val < dim_max for val, dim_max in
                                 zip(model.focus_position[1:], model.pinning_matrices[model.focus_position[0]])):
                return True
        return model.FIELD_TYPES.focus_position

    @classmethod
    def _validate_suppress_non_focal(cls, model):

        if isinstance(model.suppress_non_focal, bool):
            return True
        return model.FIELD_TYPES.suppress_non_focal

    @classmethod
    def _validate_animate_focal(cls, model):

        if isinstance(model.animate_focal, bool):
            return True
        return model.FIELD_TYPES.animate_focal

    @classmethod
    def _validate_grid_images(cls, model):

        if model.grid_images is None or (
                    cls._is_tuple_or_list(model.grid_images) and
                    all(isinstance(val, int) and 0 <= val for val in model.grid_images)):
            return True
        return model.FIELD_TYPES.grid_images

    @classmethod
    def _validate_grid_correction(cls, model):

        def _valid_correction(value):

            return value is None or (len(value) == 2 and all(isinstance(offset, int) for offset in value))

        try:
            if (all(_valid_correction(plate) for plate in model.grid_correction) and
                        len(model.grid_correction) == len(model.pinning_matrices)):
                return True
        except (TypeError, IndexError):
            pass

        return model.FIELD_TYPES.grid_correction

    @classmethod
    def _validate_grid_model(cls, model):

        if cls._is_valid_submodel(model, "grid_model"):
            return True
        return model.FIELD_TYPES.grid_model

    @classmethod
    def _validate_xml_model(cls, model):

        if cls._is_valid_submodel(model, "xml_model"):
            return True
        return model.FIELD_TYPES.xml_model


class GridModelFactory(AbstractModelFactory):
    _MODEL = GridModel
    STORE_SECTION_SERIALIZERS = {
        ('use_utso',): bool,
        ("median_coefficient",): float,
        ("manual_threshold",): float
    }

    @classmethod
    def _validate_use_utso(cls, model):

        if isinstance(model.use_utso, bool):
            return True
        return model.FIELD_TYPES.use_otsu

    @classmethod
    def _validate_median_coefficient(cls, model):

        if isinstance(model.median_coefficient, float):
            return True
        return model.FIELD_TYPES.median_coefficient

    @classmethod
    def _validate_manual_threshold(cls, model):

        if isinstance(model.manual_threshold, float):
            return True
        return model.FIELD_TYPES.manual_threshold


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

        if (cls._is_tuple_or_list(model.exclude_compartments) and
                all(compartment in COMPARTMENTS for compartment in model.exclude_compartments)):
            return True
        return model.FIELD_TYPES.exclude_compartments

    @classmethod
    def _validate_exclude_measures(cls, model):

        if (cls._is_tuple_or_list(model.exclude_measures) and
                all(measure in MEASURES for measure in model.exclude_measures)):
            return True
        return model.FIELD_TYPES.exclude_measures

    @classmethod
    def _validate_make_short_tag_version(cls, model):

        if isinstance(model.make_short_tag_version, bool):
            return True
        return model.FIELD_TYPES.make_short_tag_version

    @classmethod
    def _validate_short_tag_measure(cls, model):

        if model.short_tage_measure in MEASURES:
            return True
        return model.FIELD_TYPES.short_tag_measure