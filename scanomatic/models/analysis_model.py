from enum import Enum
import scanomatic.generics.model as model
import scanomatic
import re


class DefaultPinningFormats(Enum):

    Format_96__8x12 = (8, 12)
    """:type: DefaultPinningFormats"""
    Format_384__16x24 = (16, 24)
    """:type: DefaultPinningFormats"""
    Format_1536__32x48 = (32, 48)
    """:type: DefaultPinningFormats"""
    Format_6144__64x96 = (64, 96)
    """:type: DefaultPinningFormats"""

    def human_readable(self):

        return re.sub(r'Format_(\d+)__(\d+)x(\d+)', r'\2 x \3 (\1)', self.name)


class IMAGE_ROTATIONS(Enum):
    Landscape = 0
    Portrait = 1
    Unknown = 2


class COMPARTMENTS(Enum):
    Total = 0
    Background = 1
    Blob = 2


class MEASURES(Enum):
    Count = 0
    Sum = 1
    Mean = 2
    Median = 3
    Perimeter = 4
    IQR = 5
    IQR_Mean = 6
    Centroid = 7


class VALUES(Enum):
    Pixels = 0
    Grayscale_Targets = 1
    Cell_Estimates = 2


class AnalysisModel(model.Model):

    def __init__(
            self, compilation="",
            compile_instructions="",
            pinning_matrices=((32, 48), (32, 48), (32, 48), (32, 48)),
            use_local_fixture=False,
            email="",
            stop_at_image=-1,
            output_directory="analysis",
            focus_position=None,
            suppress_non_focal=False,
            animate_focal=False,
            one_time_positioning=True,
            one_time_grayscale=False,
            grid_images=None,
            grid_model=None,
            image_data_output_item=COMPARTMENTS.Blob,
            image_data_output_measure=MEASURES.Sum,
            chain=True,
            plate_image_inclusion=None,
            cell_count_calibration=None,
            cell_count_calibration_id=None):

        if grid_model is None:
            grid_model = GridModel()

        self.cell_count_calibration = cell_count_calibration
        self.cell_count_calibration_id = cell_count_calibration_id
        self.compilation = compilation
        self.compile_instructions = compile_instructions
        self.pinning_matrices = pinning_matrices
        self.use_local_fixture = use_local_fixture
        self.email = email
        self.stop_at_image = stop_at_image
        self.output_directory = output_directory
        self.focus_position = focus_position
        self.suppress_non_focal = suppress_non_focal
        self.animate_focal = animate_focal
        self.one_time_positioning = one_time_positioning
        self.one_time_grayscale = one_time_grayscale
        self.grid_images = grid_images
        self.grid_model = grid_model
        self.image_data_output_item = image_data_output_item
        self.image_data_output_measure = image_data_output_measure
        self.chain = chain
        self.plate_image_inclusion = plate_image_inclusion
        super(AnalysisModel, self).__init__()


class GridModel(model.Model):

    def __init__(
            self,
            use_utso=True,
            median_coefficient=0.99,
            manual_threshold=0.05,
            grid=None,
            gridding_offsets=None,
            reference_grid_folder=None):

        self.use_utso = use_utso
        self.median_coefficient = median_coefficient
        self.manual_threshold = manual_threshold
        self.grid = grid
        self.gridding_offsets = gridding_offsets
        self.reference_grid_folder = reference_grid_folder

        super(GridModel, self).__init__()


class AnalysisMetaData(model.Model):

    def __init__(
            self,
            start_time=0,
            name="",
            description="",
            interval=20.0,
            images=0,
            uuid="",
            fixture="",
            scanner="",
            project_id="",
            scanner_layout_id="",
            version=scanomatic.__version__,
            pinnings=()):

        self.start_time = start_time
        self.name = name
        self.description = description
        self.interval = interval
        self.images = images
        self.uuid = uuid
        self.fixture = fixture
        self.scanner = scanner
        self.project_id = project_id
        self.scanner_layout_id = scanner_layout_id
        self.version = version
        self.pinnings = pinnings

        super(AnalysisMetaData, self).__init__()


class AnalysisFeatures(model.Model):

    def __init__(self, index=-1, data=None, shape=tuple()):

        self.data = data
        self.shape = shape
        self.index = index

        super(AnalysisFeatures, self).__init__()
