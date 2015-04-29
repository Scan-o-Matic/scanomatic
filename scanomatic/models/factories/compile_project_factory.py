__author__ = 'martin'

import os

from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.models import compile_project_model


class CompileImageFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileImageModel

    STORE_SECTION_SERIALIZERS = {
        ('index',): int,
        ('time_stamp',): float,
        ('path',): str
    }
    STORE_SECTION_HEAD = ("index",)

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


class CompileProjectFactory(AbstractModelFactory):

    MODEL = compile_project_model.CompileInstructionsModel
    STORE_SECTION_HEAD = ("scan_model", "project_name")
    _SUB_FACTORIES = {
        compile_project_model.CompileInstructionsModel: CompileImageFactory,
    }

    STORE_SECTION_SERIALIZERS = {
        ('compile_action',): compile_project_model.COMPILE_ACTION,
        ('images',): list,
        ('path',): str,
        ('start_condition',): str,
        ('fixture',): str
    }

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