__author__ = 'martin'

from enum import Enum
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory, CompileProjectFactory

FIRST_PASS_SORTING = Enum("FIRST_PASS_SORTING", names=("Index", "Time"))


class CompilationResults(object):

    def __init__(self, compilation_path=None, compile_instructions_path=None, sort_mode=FIRST_PASS_SORTING.Time):

        self._compilation_path = compilation_path
        self._compile_instructions = None
        self._plates = None
        self._plate_position_keys = None
        self._image_models = []
        self._used_models = []
        self._loading_length = 0
        if compile_instructions_path:
            self._load_compile_instructions(compile_instructions_path)
        if compilation_path:
            self._load_compilation(self._compilation_path, sort_mode=sort_mode)

    @classmethod
    def create_from_data(cls, path, compile_instructions, image_models, used_models=None):

        if used_models is None:
            used_models = []

        new = cls()
        new._compilation_path = path
        new._compile_instructions = CompileProjectFactory.copy(compile_instructions)
        new._image_models = CompileImageAnalysisFactory.copy_iterable_of_model(list(image_models))
        new._used_models = CompileImageAnalysisFactory.copy_iterable_of_model(list(used_models))
        new._loading_length = len(new._image_models)
        return new

    def _load_compile_instructions(self, path):

        try:
            self._compile_instructions = CompileProjectFactory.serializer.load(path).next()
        except StopIteration:
            self._compile_instructions = None

    def _load_compilation(self, path, sort_mode=FIRST_PASS_SORTING.Time):

        images = CompileImageAnalysisFactory.serializer.load(path)

        if sort_mode is FIRST_PASS_SORTING.Time:
            self._image_models = list(CompileImageAnalysisFactory.copy_iterable_of_model_update_indices(images))
        else:
            self._image_models = list(CompileImageAnalysisFactory.copy_iterable_of_model_update_time(images))

        self._loading_length = len(self._image_models)

    def __len__(self):

        return self._loading_length

    def __getitem__(self, item):

        if item < 0:
            item %= len(self._image_models)
        return sorted(self._image_models, key=lambda x: x.time)[item]

    def __add__(self, other):

        """

        :type other: CompilationResults
        """

        # TODO: start time needed to add compilation results in relevant manner
        start_time_difference = 0

        other_start_index = len(self)
        other_image_models = []
        for index in range(len(other)):
            model = CompileImageAnalysisFactory.copy(other[index])
            model.time += start_time_difference
            model.index += other_start_index
            other_image_models.append(model)

        other_image_models += self._image_models
        other_image_models = sorted(other_image_models, key=lambda x: x.time)

        return CompilationResults.create_from_data(self._compilation_path, self._compile_instructions,
                                                   other_image_models, self._used_models)

    @property
    def compile_instructions(self):

        """


        :rtype: scanomatic.models.compile_project_model.CompileInstructionsModel
        """

        return self._compile_instructions

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