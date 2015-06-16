__author__ = 'martin'

import scanomatic.io.project_log as project_log
from enum import Enum
from scanomatic.models.factories.analysis_factories import MetaDataFactory, AnalysisModelFactory
from scanomatic.models.factories.fixture_factories import FixtureFactory

FIRST_PASS_SORTING = Enum("FIRST_PASS_SORTING", names=("Index", "Time"))


class CompilationResults(object):

    def __init__(self, compilation_path, compile_instructions_path, sort_mode=FIRST_PASS_SORTING.Time):

        self._file_path = path
        self._meta_data = None
        self._plates = None
        self._plate_position_keys = None
        self._image_models = []
        self._used_models = []
        self._loading_length = 0
        if path:
            self._load_path(self._file_path, sort_mode=sort_mode)

    @classmethod
    def create_from_data(cls, path, meta_data, image_models, used_models=None):

        if used_models is None:
            used_models = []

        new = cls()
        new._file_path = path
        new._meta_data = MetaDataFactory.copy(meta_data)
        new._image_models = [FixtureFactory.copy(model) for model in image_models]
        new._used_models = [FixtureFactory.copy(model) for model in used_models]
        new._loading_length = len(new._image_models)
        return new

    def _load_path(self, path, sort_mode=FIRST_PASS_SORTING.Time):

        self._meta_data = MetaDataFactory.create(**project_log.get_meta_data(path))
        if sort_mode is FIRST_PASS_SORTING.Time:
            self._image_models = list(
                FixtureFactory.create_many_update_indices(project_log.get_image_entries(path))
            )
        else:
            self._image_models = list(
                FixtureFactory.create_many_update_times(project_log.get_image_entries(path))
            )

        self._loading_length = len(self._image_models)

    def __len__(self):

        return self._loading_length

    def __getitem__(self, item):

        if item < 0:
            item %= len(self._image_models)
        return sorted(self._image_models, key=lambda x: x.time)[item]

    def __add__(self, other):

        start_time_difference = other.meta_data.start_time - self.meta_data.start_time
        other_start_index = len(self)
        other_image_models = []
        for index in range(len(other)):
            model = FixtureFactory.copy(other[index])
            model.time += start_time_difference
            model.index += other_start_index
            other_image_models.append(model)

        other_image_models += self._image_models
        other_image_models = sorted(other_image_models, key=lambda x: x.time)

        return CompilationResults.create_from_data(self._file_path, self._meta_data, other_image_models,
                                                 self._used_models)

    @property
    def meta_data(self):

        return self._meta_data

    @property
    def plates(self):

        return self[-1].plates

    @property
    def last_index(self):

        return len(self._image_models) - 1

    def recycle(self):

        self._image_models += self._used_models
        self._used_models = []

    def get_next_image_model(self):

        model = self[-1]
        self._image_models.remove(model)
        self._used_models.append(model)
        return model