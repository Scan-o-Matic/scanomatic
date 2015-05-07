__author__ = 'martin'

from scanomatic.generics.abstract_model_factory import AbstractModelFactory
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

        def _replace(key, new_key_pattern, new_key_index_names):
            if key in settings:

                for index, new_key_index_name in enumerate(new_key_index_names):
                    try:
                        settings[new_key_pattern.format(new_key_index_name)] = settings[key][index]
                    except (IndexError, TypeError):
                        pass

                del settings[key]

        _replace("center", "center_{0}", ("x", "y"))
        _replace("delta", "delta_{0}", ("x", "y"))

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