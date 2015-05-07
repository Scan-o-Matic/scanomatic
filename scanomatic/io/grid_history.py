__author__ = 'martin'

from scanomatic.models.factories.fixture_factories import GridHistoryFactory
from collections import defaultdict
from scanomatic.io.logger import Logger
import numpy as np
from scanomatic.io.paths import Paths

#
# CLASSES
#


class GriddingHistory(object):
    """This class keeps track of the gridding-histories of the fixture
    using the configuration-file in the fixtures-directory"""

    plate_pinning_pattern = "plate_{0}_pinning_{1}"
    pinning_formats = ((8, 12), (16, 24), (32, 48), (64, 96))
    plate_area_pattern = "plate_{0}_area"

    def __init__(self, fixture_settings):
        """
        :type fixture_settings: scanomatic.io.fixtures.Fixture_Settings
        """
        self._logger = Logger("Gridding History {0}".format(fixture_settings.model.name))
        self._fixture_settings = fixture_settings
        self._models_per_plate_pinning = defaultdict(dict)

    def load(self):

        self._models_per_plate_pinning.clear()
        history = GridHistoryFactory.serializer.load(self.path)
        for model in history:
            """:type : scanomatic.models.analysis_model.GridHistoryModel"""
            self._models_per_plate_pinning[(model.plate, model.pinning)].append(model)

    @property
    def path(self):

        return Paths().fixture_grid_history_pattern.format(self._fixture_settings.path)

    @property
    def _name(self):
        return self._fixture_settings.model.name

    def _get_gridding_history(self, plate, pinning_format):

        return [(model.offset_x, model.offset_y, model.delta_x, model.delta_y) for model in
                self._models_per_plate_pinning[(plate, pinning_format)].values()]

    def get_history_model(self, project_id, plate, pinning):

        models = [model for model in self._models_per_plate_pinning[(plate, pinning)]]

        if project_id in models:
            return models[project_id]

        self._logger.warning(
            "No history exists for project {0} plate {1} pinnig {2}".format(project_id, plate, pinning))
        return None

    def get_gridding_history(self, plate, pinning_format):

        history = self._get_gridding_history(plate, pinning_format)

        if not history:
            self._logger.info(
                "No history in {2} on plate {0} format {1}".format(
                    plate, pinning_format, self._name))

            return None

        self._logger.info(
            "Returning history for {0} plate {1} format {2}".format(
                self._name, plate, pinning_format))
        return np.array(history)

    def set_gridding_parameters(self, project_id, pinning_format, plate,
                                center, spacings):

        model = GridHistoryFactory.create(project_id=project_id, pinnig=pinning_format,
                                          plate=plate, center=center, delta=spacings)

        if GridHistoryFactory.validate(model):

            if not GridHistoryFactory.serializer.dump(model, self.path):

                self._logger.warning("Could not write grid history")
                return False
        else:

            self._logger.warning("This is not a valid grid history")
            return False

        self._logger.info("Setting history {0} on fixture {1} for {2} {3}".format(
            center + spacings, self._name, project_id, plate))

        self._models_per_plate_pinning[(model.plate, model.pinning)][model.project_id] = model
        return True

    def unset_gridding_parameters(self, project_id, pinning_format, plate):

        model = self.get_history_model(project_id, plate, pinning_format)
        if model:
            GridHistoryFactory.serializer.purge(model, self.path)

    def reset_gridding_history(self, plate):

        for plate_in_history, pin_format in self._models_per_plate_pinning:
            if plate == plate_in_history:
                for model in self._models_per_plate_pinning[(plate_in_history, pin_format)]:
                    GridHistoryFactory.serializer.purge(model, self.path)
            del self._models_per_plate_pinning[(plate_in_history, pin_format)]

    def reset_all_gridding_histories(self):

        GridHistoryFactory.serializer.purge_all(self.path)
        self._models_per_plate_pinning.clear()