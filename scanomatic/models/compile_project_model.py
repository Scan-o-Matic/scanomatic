from enum import Enum
from scanomatic.generics.model import Model


class COMPILE_ACTION(Enum):

    Initiate = 0
    Append = 1
    InitiateAndSpawnAnalysis = 10
    AppendAndSpawnAnalysis = 11


class FIXTURE(Enum):

    Local = 0
    Global = 1


class CompileInstructionsModel(Model):

    def __init__(self, compile_action=COMPILE_ACTION.InitiateAndSpawnAnalysis,
                 start_time=0.0, images=tuple(), path="", start_condition="",
                 fixture_type=FIXTURE.Local, fixture_name=None, email="",
                 overwrite_pinning_matrices=None,
                 cell_count_calibration_id="default"):

        self.compile_action = compile_action
        self.images = images
        self.path = path
        self.start_time = start_time
        self.start_condition = start_condition
        self.fixture_type = fixture_type
        self.fixture_name = fixture_name
        self.email = email
        self.overwrite_pinning_matrices = overwrite_pinning_matrices
        self.cell_count_calibration_id = cell_count_calibration_id

        super(CompileInstructionsModel, self).__init__()


class CompileImageModel(Model):

    def __init__(self, index=-1, path="", time_stamp=0.0):

        self.index = index
        self.path = path
        self.time_stamp = time_stamp

        super(CompileImageModel, self).__init__()


class CompileImageAnalysisModel(Model):

    def __init__(self, image=None, fixture=None):

        self.image = image
        """:type : CompileImageModel"""
        self.fixture = fixture
        """:type : scanomatic.models.fixture_models.FixtureModel"""

        super(CompileImageAnalysisModel, self).__init__()
