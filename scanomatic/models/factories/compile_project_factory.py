import os
import re
import glob

from scanomatic.generics.abstract_model_factory import AbstractModelFactory, email_serializer
from scanomatic.models import compile_project_model
from scanomatic.models import fixture_models
from scanomatic.models.factories import fixture_factories
from scanomatic.io.paths import Paths
from scanomatic.io.fixtures import Fixtures
from scanomatic.data_processing.calibration import get_active_cccs


class CompileImageFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileImageModel

    STORE_SECTION_SERIALIZERS = {
        'index': int,
        'time_stamp': float,
        'path': str
    }
    STORE_SECTION_HEAD = ("index",)

    @classmethod
    def create(cls, **settings):

        """

        :rtype : scanomatic.models.compile_project_model.CompileImageModel
        """
        model = super(CompileImageFactory, cls).create(**settings)

        if not model.time_stamp:
            cls.set_time_stamp_from_path(model)

        if model.index < 0:
            cls.set_index_from_path(model)

        return model

    @classmethod
    def _validate_index(cls, model):

        """
        :type model: scanomatic.models.compile_project_model.CompileImageModel
        """
        if model.index >= 0:
            return True
        return model.FIELD_TYPES.index

    @classmethod
    def _validate_path(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileImageModel
        """
        if os.path.abspath(model.path) == model.path and os.path.isfile(model.path):
            return True
        return model.FIELD_TYPES.path

    @classmethod
    def _validate_time_stamp(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileImageModel
        """
        if model.time_stamp >= 0.0:

            return True
        return model.FIELD_TYPES.time_stamp

    @classmethod
    def set_time_stamp_from_path(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileImageModel
        """
        match = re.search(r'(\d+\.\d+)\.tiff$', model.path)
        if match:
            model.time_stamp = float(match.groups()[0])

    @classmethod
    def set_index_from_path(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileImageModel
        """
        match = re.search(r'_(\d+)_(\d+\.\d+)\.tiff$', model.path)
        if match:
            model.index = int(match.groups()[0])


class CompileProjectFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileInstructionsModel
    STORE_SECTION_HEAD = ("path",)
    _SUB_FACTORIES = {
        compile_project_model.CompileImageModel: CompileImageFactory,
    }

    STORE_SECTION_SERIALIZERS = {
        'compile_action': compile_project_model.COMPILE_ACTION,
        'images': (tuple, compile_project_model.CompileImageModel),
        'path': str,
        'start_condition': str,
        'email': email_serializer,
        'start_time': float,
        'fixture_type': compile_project_model.FIXTURE,
        'fixture_name': str,
        'overwrite_pinning_matrices': (tuple, tuple, int),
        'cell_count_calibration_id': str,
    }

    @classmethod
    def create(cls, **settings):
        """
        :rtype : scanomatic.models.compile_project_model.CompileInstructionsModel
        """

        model = super(CompileProjectFactory, cls).create(**settings)
        return model

    @classmethod
    def dict_from_path_and_fixture(cls, path, fixture=None, is_local=None,
                                   compile_action=compile_project_model.COMPILE_ACTION.Initiate, **kwargs):

        """

        :type path: str
        """
        path = path.rstrip("/")

        if path != os.path.abspath(path):
            cls.logger.error("Not an absolute path, aborting")
            return {}

        if is_local is None:
            is_local = not fixture

        image_path = os.path.join(path, "*.tiff")

        images = [{'path': p, 'index': i} for i, p in enumerate(sorted(glob.glob(image_path)))]

        return cls.to_dict(cls.create(
            compile_action=compile_action.name,
            images=images,
            fixture_type=
                is_local and compile_project_model.FIXTURE.Local.name or compile_project_model.FIXTURE.Global.name,
            fixture_name=fixture,
            path=path,
            **kwargs))

    @classmethod
    def _validate_images(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileInstructionsModel
        """
        try:
            for image in model.images:
                if not CompileImageFactory.validate(image):
                    return model.FIELD_TYPES.images
            if model.images:
                return True
            else:
                return model.FIELD_TYPES.images
        except (IndexError, TypeError):
            return model.FIELD_TYPES.images

    @classmethod
    def _validate_path(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileInstructionsModel
        """
        basename = os.path.basename(model.path)
        dirname = os.path.dirname(model.path)

        if model.path != dirname and os.path.isdir(dirname) and os.path.abspath(dirname) == dirname and basename:
            return True
        return model.FIELD_TYPES.path

    @classmethod
    def _validate_fixture(cls, model):
        """
        :type model: scanomatic.models.compile_project_model.CompileInstructionsModel
        """
        if model.fixture_type is compile_project_model.FIXTURE.Local:
            if os.path.isfile(os.path.join(model.path, Paths().experiment_local_fixturename)):
                return True
            else:
                return model.FIELD_TYPES.fixture_type
        elif model.fixture_type is compile_project_model.FIXTURE.Global:
            if model.fixture_name in Fixtures():
                return True
            else:
                return model.FIELD_TYPES.fixture_name
        else:
            return model.FIELD_TYPES.fixture_type

    @classmethod
    def _validate_cell_count_calibration_id(cls, model):
        """

        :type model: scanomatic.models.scanning_model.ScanningModel
        """
        if model.cell_count_calibration_id in get_active_cccs():
            return True
        return model.FIELD_TYPES.cell_count_calibration


class CompileImageAnalysisFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileImageAnalysisModel
    STORE_SECTION_HEAD = "Image"
    _SUB_FACTORIES = {
        compile_project_model.CompileImageModel: CompileImageFactory,
        fixture_models.FixtureModel: fixture_factories.FixtureFactory
    }

    STORE_SECTION_SERIALIZERS = {
        'image': compile_project_model.CompileImageModel,
        'fixture': fixture_models.FixtureModel
    }

    @classmethod
    def create(cls, **settings):
        """:rtype : scanomatic.models.compile_project_model.CompileImageAnalysisModel"""
        return super(CompileImageAnalysisFactory, cls).create(**settings)

    @classmethod
    def _validate_fixture(cls, model):
        """:type model : scanomatic.models.compile_project_model.CompileImageAnalysisModel"""

        if cls._is_valid_submodel(model, "fixture"):
            return True
        else:
            return model.FIELD_TYPES.fixture

    @classmethod
    def _validate_image(cls, model):
        """:type model : scanomatic.models.compile_project_model.CompileImageAnalysisModel"""

        if cls._is_valid_submodel(model, "image"):
            return True
        else:
            return model.FIELD_TYPES.image

    @classmethod
    def copy_iterable_of_model_update_indices(cls, iterable):

        models = cls.copy_iterable_of_model(iterable)
        for (index, m) in enumerate(sorted(models, key=lambda x: x.image.time_stamp)):
            m.image.index = index
            yield m

    @classmethod
    def copy_iterable_of_model_update_time(cls, iterable):

        models = cls.copy_iterable_of_model(iterable)
        inject_time = 0
        previous_time = 0
        for (index, m) in enumerate(models):
            m.index = index
            if m.time < previous_time:
                inject_time += previous_time - m.time
            m.time += inject_time
            yield m
