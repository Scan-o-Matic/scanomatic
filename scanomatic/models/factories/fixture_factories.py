import re

from scanomatic.models.fixture_models import FixtureModel, FixturePlateModel
from scanomatic.generics.abstract_model_factory import (
    AbstractModelFactory, rename_setting, float_list_serializer
)
from scanomatic.models import fixture_models


class FixturePlateFactory(AbstractModelFactory):
    MODEL = FixturePlateModel
    STORE_SECTION_HEAD = "Plate Area"
    STORE_SECTION_SERIALIZERS = {
        "index": int,
        "x1": int,
        "y1": int,
        "x2": int,
        "y2": int
    }

    @classmethod
    def create(cls, **settings):

        """

        :rtype : scanomatic.models.fixture_models.FixturePlateModel
        """
        return super(FixturePlateFactory, cls).create(**settings)


class GrayScaleAreaModelFactory(AbstractModelFactory):

    MODEL = fixture_models.GrayScaleAreaModel
    STORE_SECTION_HEAD = "Grayscale"
    STORE_SECTION_SERIALIZERS = {
        'name': str,
        'values': float_list_serializer,
        'width': float,
        'section_length': float,
        'x1': int,
        'x2': int,
        'y1': int,
        'y2': int,
    }

    @classmethod
    def create(cls, **settings):

        """

        :rtype : scanomatic.models.fixture_models.FixturePlateModel
        """
        return super(GrayScaleAreaModelFactory, cls).create(**settings)


class FixtureFactory(AbstractModelFactory):

    MODEL = FixtureModel
    STORE_SECTION_HEAD = ('name',)
    _SUB_FACTORIES = {
        fixture_models.FixturePlateModel: FixturePlateFactory,
        fixture_models.GrayScaleAreaModel: GrayScaleAreaModelFactory
    }

    STORE_SECTION_SERIALIZERS = {
        'grayscale': fixture_models.GrayScaleAreaModel,
        "orientation_marks_x": float_list_serializer,
        "orientation_marks_y": float_list_serializer,
        "shape": list,
        "coordinates_scale": float,
        "scale": float,
        "path": str,
        "name": str,
        "plates": (tuple, fixture_models.FixturePlateModel)
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
