from enum import Enum
from scanomatic.models.factories.compile_project_factory import CompileImageAnalysisFactory, CompileProjectFactory
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
from scanomatic.io.logger import Logger
from scanomatic.io.paths import Paths
import os
from glob import glob

FIRST_PASS_SORTING = Enum("FIRST_PASS_SORTING", names=("Index", "Time"))


class CompilationResults(object):

    def __init__(self, compilation_path=None, compile_instructions_path=None,
                 scanner_instructions_path=None, sort_mode=FIRST_PASS_SORTING.Time):

        self._logger = Logger("Compilation results")
        self._compilation_path = compilation_path
        self._compile_instructions = None
        self._scanner_instructions = None
        """:type : scanomatic.models.scanning_model.ScanningModel"""
        self.load_scanner_instructions(scanner_instructions_path)
        self._plates = None
        self._plate_position_keys = None
        self._image_models = []
        self._used_models = []
        self._current_model = None
        self._loading_length = 0
        if compile_instructions_path:
            self._load_compile_instructions(compile_instructions_path)
        if compilation_path:
            self._load_compilation(self._compilation_path, sort_mode=sort_mode)

    @classmethod
    def create_from_data(cls, path, compile_instructions, image_models, used_models=None, scan_instructions=None):

        if used_models is None:
            used_models = []

        new = cls()
        new._compilation_path = path
        new._compile_instructions = CompileProjectFactory.copy(compile_instructions)
        new._image_models = CompileImageAnalysisFactory.copy_iterable_of_model(list(image_models))
        new._used_models = CompileImageAnalysisFactory.copy_iterable_of_model(list(used_models))
        new._loading_length = len(new._image_models)
        new._scanner_instructions = scan_instructions
        return new

    def load_scanner_instructions(self, path=None):
        """

        Args:
            path:  Path to the instrucitons or None to infer it


        """
        if path is None:
            try:
                path = glob(os.path.join(os.path.dirname(self._compilation_path),
                                         Paths().scan_project_file_pattern.format('*')))[0]
            except IndexError:
                self._logger.warning("No information of start time of project, can't safely be joined with others")
                return

        self._scanner_instructions = ScanningModelFactory.serializer.load_first(path)

    def _load_compile_instructions(self, path):

        try:
            self._compile_instructions = CompileProjectFactory.serializer.load_first(path)
        except IndexError:
            self._logger.error("Could not load path {0}".format(path))
            self._compile_instructions = None

    def _load_compilation(self, path, sort_mode=FIRST_PASS_SORTING.Time):

        images = CompileImageAnalysisFactory.serializer.load(path)
        self._logger.info("Loaded {0} compiled images".format(len(images)))

        self._reindex_plates(images)

        if sort_mode is FIRST_PASS_SORTING.Time:
            self._image_models = list(CompileImageAnalysisFactory.copy_iterable_of_model_update_indices(images))
        else:
            self._image_models = list(CompileImageAnalysisFactory.copy_iterable_of_model_update_time(images))

        self._loading_length = len(self._image_models)

    @staticmethod
    def _reindex_plates(images):

        for image in images:

            if image and image.fixture and image.fixture.plates:

                for plate in image.fixture.plates:
                    plate.index -= 1

    def __len__(self):

        return self._loading_length

    def __getitem__(self, item):
        """


        :rtype: scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        if not self._image_models:
            return None

        if item < 0:
            item %= len(self._image_models)

        try:
            return sorted(self._image_models, key=lambda x: x.image.time_stamp)[item]
        except (ValueError, IndexError):
            return None

    def keys(self):

        if self._image_models is None:
            return []
        return range(len(self._image_models))

    def __add__(self, other):

        """

        :type other: CompilationResults
        """

        start_time_difference = other.start_time - self.start_time

        other_start_index = len(self)
        other_image_models = []
        other_directory = os.path.dirname(other._compilation_path)
        for index in range(len(other)):
            model = CompileImageAnalysisFactory.copy(other[index])
            """:type : scanomatic.models.compile_project_model.CompileImageAnalysisModel"""

            model.image.time_stamp += start_time_difference
            model.image.index += other_start_index
            self._update_image_path_if_needed(model, other_directory)
            other_image_models.append(model)

        other_image_models += self._image_models
        other_image_models = sorted(other_image_models, key=lambda x: x.image.time_stamp)

        return CompilationResults.create_from_data(self._compilation_path, self._compile_instructions,
                                                   other_image_models, self._used_models, self._scanner_instructions)

    def _update_image_path_if_needed(self, model, directory):
        if not os.path.isfile(model.image.path):
            image_name = os.path.basename(model.image.path)
            if os.path.isfile(os.path.join(directory, image_name)):
                model.image.path = os.path.join(directory, image_name)
                return
        self._logger.warning("Couldn't locate the file {0}".format(model.image.path))

    @property
    def start_time(self):

        if self._scanner_instructions:
            return self._scanner_instructions.start_time
        self._logger.warning("No scanner instructions have been loaded, start time unknown")
        return 0

    @property
    def compile_instructions(self):

        """


        :rtype: scanomatic.models.compile_project_model.CompileInstructionsModel
        """

        return self._compile_instructions

    @property
    def plates(self):

        res = self[-1]
        if res:
            return res.fixture.plates
        return None

    @property
    def last_index(self):

        return len(self._image_models) - 1

    @property
    def total_number_of_images(self):

        return len(self._image_models) + len(self._used_models)

    @property
    def current_image(self):
        """

        :rtype : scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        return self._current_model

    @property
    def current_absolute_time(self):

        return self.current_image.image.time_stamp + self.compile_instructions.start_time

    def recycle(self):

        self._image_models += self._used_models
        self._used_models = []
        self._current_model = None

    def get_next_image_model(self):
        """

        :rtype : scanomatic.models.compile_project_model.CompileImageAnalysisModel
        """
        model = self[-1]
        self._current_model = model
        if model:
            self._image_models.remove(model)
            self._used_models.append(model)
        return model

    def dump(self, directory, new_name=None, force_dump_scan_instructions=False):

        self._logger.warning(
            """This functionality has not fully been tested, if you test it and it works fine let Martin konw.
            If it doesn't work, let him know too.""")
        directory = os.path.abspath(directory)
        os.makedirs(directory)
        if new_name is None:
            new_name = os.path.basename(directory)

        try:
            with open(os.path.join(directory, Paths().project_compilation_pattern.format(new_name)), 'w') as fh:
                while True:
                    model = CompileImageAnalysisFactory.copy(self.get_next_image_model())
                    self._update_image_path_if_needed(model, directory)
                    if model is None:
                        break
                    CompileImageAnalysisFactory.serializer.dump_to_filehandle(model, fh)
        except IOError:
            self._logger.error("Could not save to directory")
            return

        compile_instructions = os.path.join(directory, Paths().project_compilation_pattern.format(new_name))
        CompileProjectFactory.serializer.dump(self._compile_instructions, compile_instructions)

        if not glob(os.path.join(directory, Paths().scan_project_file_pattern.format('*'))) or \
                force_dump_scan_instructions:

            scan_instructions = os.path.join(directory, Paths().scan_project_file_pattern.format(new_name))
            ScanningModelFactory.serializer.dump(self._scanner_instructions, scan_instructions)