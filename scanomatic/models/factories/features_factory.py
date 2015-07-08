__author__ = 'martin'

import os

from scanomatic.generics.abstract_model_factory import AbstractModelFactory


class FeaturesFactory(AbstractModelFactory):

    STORE_SECTION_SERIALIZERS = {
        "analysis_directory": str
    }

    @classmethod
    def _validate_analysis_directory(cls, model):

        if not isinstance(model.analysis_directory, str):
            return model.FIELD_TYPES.analysis_directory

        analysis_directory = model.analysis_directory.rstrip("/")
        if (os.path.abspath(analysis_directory) == analysis_directory and
                os.path.isdir(model.analysis_directory)):

            return True
        return model.FIELD_TYPES.analysis_directory