__author__ = 'martin'

import os
import re

from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.models import compile_project_model
from scanomatic.io.paths import Paths
from scanomatic.io.fixtures import Fixtures


class CompileImageFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileImageModel

    STORE_SECTION_SERIALIZERS = {
        ('index',): int,
        ('time_stamp',): float,
        ('path',): str
    }
    STORE_SECTION_HEAD = ("index",)

    @classmethod
    def create(cls, **settings):

        model = super(CompileImageFactory, cls).create(**settings)

        if not model.time_stamp:
            cls.set_time_stamp_from_path(model)

        if model.index < 0:
            cls.set_index_from_path(model)

        return model

    @classmethod
    def _validate_index(cls, model):

        if model.index >= 0:
            return True
        return model.FIELD_TYPES.index

    @classmethod
    def _validate_path(cls, model):

        if os.path.abspath(model.path) == model.path and os.path.isfile(model.path):
            return True
        return model.FIELD_TYPES.path

    @classmethod
    def _validate_time_stamp(cls, model):

        if model.time_stamp >= 0.0:

            return True
        return model.FIELD_TYPES.time_stamp

    @classmethod
    def set_time_stamp_from_path(cls, model):

        match = re.search(r'(\d+\.\d+)\.tiff$', model.path)
        if match:
            model.time_stamp = float(match.groups()[0])

    @classmethod
    def set_index_from_path(cls, model):

        match = re.search(r'_(\d+)_(\d+\.\d+)\.tiff$', model.path)
        if match:
            model.index = int(match.groups()[0])


class CompileProjectFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileInstructionsModel
    STORE_SECTION_HEAD = ("scan_model", "project_name")
    _SUB_FACTORIES = {
        compile_project_model.CompileImageModel: CompileImageFactory,
    }

    STORE_SECTION_SERIALIZERS = {
        ('compile_action',): compile_project_model.COMPILE_ACTION,
        ('images',): list,
        ('path',): str,
        ('start_condition',): str,
        ('fixture',): compile_project_model.FIXTURE,
        ('fixture_name',): str
    }

    @classmethod
    def create(cls, **settings):

        model = super(CompileProjectFactory, cls).create(**settings)
        cls.enforce_subfactory_list(model)
        return model

    @classmethod
    def enforce_subfactory_list(cls, model):

        for i in range(len(model.images)):

            if not isinstance(model.images[i], compile_project_model.CompileImageModel):
                model.images[i] = CompileImageFactory.create(**model.images[i])

    @classmethod
    def _validate_images(cls, model):

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

        basename = os.path.basename(model.path)
        dirname = os.path.dirname(model.path)

        if model.path != dirname and os.path.isdir(dirname) and os.path.abspath(dirname) == dirname and basename:
            return True
        return model.FIELD_TYPES.path

    @classmethod
    def _validate_fixture(cls, model):

        if model.fixture is compile_project_model.FIXTURE.Local:
            if os.path.isfile(os.path.join(model.path, Paths().experiment_local_fixturename)):
                return True
            else:
                return model.FIELD_TYPES.fixture
        elif model.fixture is compile_project_model.FIXTURE.Global:
            if model.fixture_name in Fixtures():
                return True
            else:
                return model.FIELD_TYPES.fixture_name
        else:
            return model.FIELD_TYPES.fixture