__author__ = 'martin'

import os

from scanomatic.generics.abstract_model_factory import AbstractModelFactory, rename_setting
import scanomatic.models.analysis_model as analysis_model


class GridModelFactory(AbstractModelFactory):
    MODEL = analysis_model.GridModel
    STORE_SECTION_SERIALIZERS = {
        ('use_utso',): bool,
        ("median_coefficient",): float,
        ("manual_threshold",): float,
        ("gridding_offsets",): list
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

    @classmethod
    def _validate_grid_offsets(cls, model):

        def _valid_correction(value):

            return value is None or (len(value) == 2 and all(isinstance(offset, int) for offset in value))

        if model.gridding_offsets is None:
            return True

        try:
            if all(_valid_correction(plate) for plate in model.gridding_offsets):
                return True
        except (TypeError, IndexError):
            pass

        return model.FIELD_TYPES.gridding_offsets


class XMLModelFactory(AbstractModelFactory):
    MODEL = analysis_model.XMLModel
    STORE_SECTION_SERIALIZERS = {
        ("exclude_compartments",): tuple,
        ("exclude_measures",): tuple,
        ("make_short_tag_version",): bool,
        ("short_tag_measure",): analysis_model.MEASURES
    }

    @classmethod
    def _validate_exclude_compartments(cls, model):

        if (cls._is_tuple_or_list(model.exclude_compartments) and
                all(compartment in analysis_model.COMPARTMENTS for compartment in model.exclude_compartments)):
            return True
        return model.FIELD_TYPES.exclude_compartments

    @classmethod
    def _validate_exclude_measures(cls, model):

        if (cls._is_tuple_or_list(model.exclude_measures) and
                all(measure in analysis_model.MEASURES for measure in model.exclude_measures)):
            return True
        return model.FIELD_TYPES.exclude_measures

    @classmethod
    def _validate_make_short_tag_version(cls, model):

        if isinstance(model.make_short_tag_version, bool):
            return True
        return model.FIELD_TYPES.make_short_tag_version

    @classmethod
    def _validate_short_tag_measure(cls, model):

        if model.short_tag_measure in analysis_model.MEASURES:
            return True
        return model.FIELD_TYPES.short_tag_measure


class AnalysisModelFactory(AbstractModelFactory):
    MODEL = analysis_model.AnalysisModel
    STORE_SECTION_HEAD = ("first_pass_file",)
    _SUB_FACTORIES = {
        analysis_model.XMLModel: XMLModelFactory,
        analysis_model.GridModel: GridModelFactory
    }

    STORE_SECTION_SERIALIZERS = {
        ('compilation',): str,
        ('compile_instructions',): str,
        ('pinning_matrices',): list,
        ('use_local_fixture',): bool,
        ('stop_at_image',): int,
        ('output_directory',): str,
        ('focus_position',): tuple,
        ('suppress_non_focal',): bool,
        ('animate_focal',): bool,
        ('grid_images',): list,
        ('grid_model',): GridModelFactory,
        ('xml_model',): XMLModelFactory
    }

    @classmethod
    def create(cls, **settings):
        """


        :rtype : scanomatic.models.analysis_model.AnalysisModel
        """
        return super(cls, AnalysisModelFactory).create(**settings)


    @classmethod
    def set_absolute_paths(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """

        base_path = os.path.dirname(model.compilation)
        model.analysis_config_file = cls._get_absolute_path(model.analysis_config_file, base_path)
        model.output_directory = cls._get_absolute_path(model.output_directory, base_path)

    @classmethod
    def _get_absolute_path(cls, path, base_path):

        if os.path.abspath(path) != path:
            return os.path.join(base_path, path)
        return path

    @classmethod
    def _validate_compilation_file(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if cls._is_file(model.compilation) and os.path.abspath(model.compilation) == model.compilation:
            return True
        return model.FIELD_TYPES.compilation

    @classmethod
    def _validate_compilation_instructions_file(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if model.compile_instructions in (None, "") or AbstractModelFactory._is_file(model.compile_instructions):
            return True
        return model.FIELD_TYPES.analysis_config_file

    @classmethod
    def _validate_pinning_matrices(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if AbstractModelFactory._is_pinning_formats(model.pinning_matrices):
            return True
        return model.FIELD_TYPES.pinning_matrices

    @classmethod
    def _validate_use_local_fixture(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if isinstance(model.use_local_fixture, bool):
            return True
        return model.FIELD_TYPES.use_local_fixture

    @classmethod
    def _validate_stop_at_image(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if isinstance(model.stop_at_image, int):
            return True
        return model.FIELD_TYPES.stop_at_image

    @classmethod
    def _validate_output_directory(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if (model.output_directory is None or isinstance(model.output_directory, str) and
                os.sep not in model.output_directory):
            return True
        return model.FIELD_TYPES.output_directory

    @classmethod
    def _validate_focus_position(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
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
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if isinstance(model.suppress_non_focal, bool):
            return True
        return model.FIELD_TYPES.suppress_non_focal

    @classmethod
    def _validate_animate_focal(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if isinstance(model.animate_focal, bool):
            return True
        return model.FIELD_TYPES.animate_focal

    @classmethod
    def _validate_grid_images(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if model.grid_images is None or (
                cls._is_tuple_or_list(model.grid_images) and
                all(isinstance(val, int) and 0 <= val for val in model.grid_images)):
            return True
        return model.FIELD_TYPES.grid_images

    @classmethod
    def _validate_grid_model(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if cls._is_valid_submodel(model, "grid_model"):
            return True
        return model.FIELD_TYPES.grid_model

    @classmethod
    def _validate_xml_model(cls, model):
        """

        :type model: scanomatic.models.analysis_model.AnalysisModel
        """
        if cls._is_valid_submodel(model, "xml_model"):
            return True
        return model.FIELD_TYPES.xml_model


class MetaDataFactory(AbstractModelFactory):
    MODEL = analysis_model.AnalysisMetaData
    STORE_SECTION_SERIALIZERS = {
        ("start_time",): float,
        ("name",): str,
        ("description",): str,
        ("interval",): float,
        ("images",): int,
        ("uuid",): str,
        ("fixture", ): str,
        ("scanner",): str,
        ("project_id",): str,
        ("scanner_layout_id",): str,
        ("version",): float,
        ("pinnings",): list
    }

    @classmethod
    def create(cls, **settings):

        for (old_name, new_name) in [
                ("Start Time", "start_time"),
                ("Prefix", "name"), ("Interval", "interval"), ("Description", "description"),
                ("Version", "version"), ("UUID", "uuid"), ("Measures", "images"), ("Fixture", "fixture"),
                ("Scanner", "scanner"), ("Pinning Matrices", "pinnings"), ("Project ID", "project_id"),
                ("Scanner Layout ID", "scanner_layout_id")]:

            rename_setting(settings, old_name, new_name)

        if "Manual Gridding" in settings:
            del settings["Manual Gridding"]

        return super(MetaDataFactory, cls).create(**settings)