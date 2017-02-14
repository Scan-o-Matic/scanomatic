import os
from types import StringTypes

from scanomatic.generics.abstract_model_factory import AbstractModelFactory, email_serializer
import scanomatic.models.features_model as features_model


class FeaturesFactory(AbstractModelFactory):

    MODEL = features_model.FeaturesModel
    STORE_SECTION_HEAD = ("analysis_directory", )
    STORE_SECTION_SERIALIZERS = {
        "analysis_directory": str,
        "email": email_serializer,
        "extraction_data": features_model.FeatureExtractionData,
        "try_keep_qc": bool
    }

    @classmethod
    def _validate_analysis_directory(cls, model):

        if not isinstance(model.analysis_directory, StringTypes):
            return model.FIELD_TYPES.analysis_directory

        analysis_directory = model.analysis_directory.rstrip("/")
        if (os.path.abspath(analysis_directory) == analysis_directory and
                os.path.isdir(model.analysis_directory)):

            return True
        return model.FIELD_TYPES.analysis_directory

    @classmethod
    def create(cls, **settings):
        """:rtype : scanomatic.models.features_model.FeaturesModel"""

        return super(FeaturesFactory, cls).create(**settings)