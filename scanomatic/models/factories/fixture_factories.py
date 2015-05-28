__author__ = 'martin'

import re

from scanomatic.models.fixture_models import FixtureModel, FixturePlateModel
from scanomatic.generics.abstract_model_factory import AbstractModelFactory, rename_setting, split_and_replace
from scanomatic.models import fixture_models


class GridHistoryFactory(AbstractModelFactory):

    MODEL = fixture_models.GridHistoryModel
    STORE_SECTION_HEAD = [('project_id',), ('plate',)]
    STORE_SECTION_SERIALIZERS = {
        ('project_id',): str,
        ('pinning',): tuple,
        ('plate',): int,
        ('center_x',): float,
        ('center_y',): float,
        ('delta_x',): float,
        ('delta_y',): float,
    }

    @classmethod
    def create(cls, **settings):

        split_and_replace(settings, "center", "center_{0}", ("x", "y"))
        split_and_replace(settings, "delta", "delta_{0}", ("x", "y"))

        return super(GridHistoryFactory, cls).create(**settings)

    @classmethod
    def _validate_plate(cls, model):

        """

        :type model: scanomatic.models.analysis_model.GridHistoryModel
        """
        if model.plate >= 0:
            return True
        else:
            return model.FIELD_TYPES.plate

    @classmethod
    def _validate_pinning(cls, model):

        """

        :type model: scanomatic.models.analysis_model.GridHistoryModel
        """
        if len(model.pinning) == 2 and all(isinstance(v, int) for v in model.pinning):
            return True
        else:
            return model.FIELD_TYPES.pinning

    @classmethod
    def _validate_delta_x(cls, model):

        """

        :type model: scanomatic.models.analysis_model.GridHistoryModel
        """
        if model.delta_x > 0:
            return True
        else:
            return model.FIELD_TYPES.delta_x

    @classmethod
    def _validate_delta_y(cls, model):

        """

        :type model: scanomatic.models.analysis_model.GridHistoryModel
        """
        if model.delta_y > 0:
            return True
        else:
            return model.FIELD_TYPES.delta_y

    @classmethod
    def _validate_center_x(cls, model):

        """

        :type model: scanomatic.models.analysis_model.GridHistoryModel
        """
        if model.center_x > 0:
            return True
        else:
            return model.FIELD_TYPES.center_x

    @classmethod
    def _validate_center_y(cls, model):

        """

        :type model: scanomatic.models.analysis_model.GridHistoryModel
        """
        if model.center_y > 0:
            return True
        else:
            return model.FIELD_TYPES.center_y


class FixtureFactory(AbstractModelFactory):

    MODEL = FixtureModel
    STORE_SECTION_HEAD = ('name',)
    STORE_SECTION_SERIALIZERS = {
        ('grayscale',): fixture_models.GrayScaleAreaModel,
        ("orientation_marks_x",): list,
        ("orientation_marks_y",): list,
        ("shape",): list,
        ("coordinates_scale",): float,
        ("scale",): float,
        ("path",): str,
        ("name",): str,
        ("plates",): list,  # TODO: This won't serialize well
    }

    @classmethod
    def create(cls, **settings):

        """

        :rtype : scanomatic.models.fixture_models.FixtureModel
        """
        plate_str = "plate_{0}_area"
        plate_index_pattern = r"\d+"

        def get_index_from_name(name):
            peoples_index_offset = 1
            return int(re.search(plate_index_pattern, name).group()) - peoples_index_offset

        for (old_name, new_name) in [("grayscale_indices", "grayscale_targets"),
                                     ("mark_X", "orientation_marks_x"),
                                     ("mark_Y", "orientation_marks_y"),
                                     ("Image Shape", "shape"), ("Scale", "coordinates_scale"),
                                     ("File", "path"), ("Time", "time")]:

            rename_setting(settings, old_name, new_name)

        if "plates" not in settings or not settings["plates"]:
            settings["plates"] = []

        for plate_name in re.findall(plate_str.format(plate_index_pattern), ", ".join(settings.keys())):
            index = get_index_from_name(plate_name)
            if plate_name in settings and settings[plate_name]:
                (x1, y1), (x2, y2) = settings[plate_name]
                settings["plates"].append(FixturePlateFactory.create(index=index, x1=x1, x2=x2, y1=y1, y2=y2))
                del settings[plate_name]

        return super(FixtureFactory, cls).create(**settings)

    @classmethod
    def create_many_update_indices(cls, iterable):

        models = [cls.create(**data) for data in iterable]
        for (index, m) in enumerate(sorted(models, key=lambda x: x.time)):
            m.index = index
            yield m

    @classmethod
    def create_many_update_times(cls, iterable):

        models = [cls.create(**data) for data in iterable]
        inject_time = 0
        previous_time = 0
        for (index, m) in enumerate(models):
            m.index = index
            if m.time < previous_time:
                inject_time += previous_time - m.time
            m.time += inject_time
            yield m


class FixturePlateFactory(AbstractModelFactory):
    MODEL = FixturePlateModel
    STORE_SECTION_SERIALIZERS = {
        ("index",): int,
        ("x1",): int,
        ("y1",): int,
        ("x2",): int,
        ("y2",): int
    }