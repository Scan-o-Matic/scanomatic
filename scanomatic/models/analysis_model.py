__author__ = 'martin'
__version__ = 0.998

from enum import Enum

import scanomatic.generics.model as model


IMAGE_ROTATIONS = Enum("IMAGE_ROTATIONS", names=("Landscape", "Portrait", "None"))

COMPARTMENTS = Enum("COMPARTMENTS", names=("Total", "Background", "Blob"))

MEASURES = Enum("MEASURES", names=("Count", "Sum", "Mean", "Median", "Perimeter", "IQR", "IQR_Mean", "Centroid"))

VALUES = Enum("Values", names=("Pixels", "Grayscale_Targets", "Cell_Estimates"))

ITEMS = Enum("ITEMS", names=("Cell", "Blob", "Background"))


class AnalysisModel(model.Model):

    def __init__(self, compilation="", compile_instructions="",
                 pinning_matrices=((32, 48), (32, 48), (32, 48), (32, 48)),
                 use_local_fixture=False,
                 stop_at_image=-1, output_directory="analysis", focus_position=None, suppress_non_focal=False,
                 animate_focal=False, grid_images=None, grid_model=None, xml_model=None,
                 image_data_output_item=ITEMS.Blob, image_data_output_measure=MEASURES.Sum):

        if grid_model is None:
            grid_model = GridModel()

        if xml_model is None:
            xml_model = XMLModel()

        self.compilation = compilation
        self.compile_instructions = compile_instructions
        self.pinning_matrices = pinning_matrices
        self.use_local_fixture = use_local_fixture
        self.stop_at_image = stop_at_image
        self.output_directory = output_directory
        self.focus_position = focus_position
        self.suppress_non_focal = suppress_non_focal
        self.animate_focal = animate_focal
        self.grid_images = grid_images
        self.grid_model = grid_model
        self.xml_model = xml_model
        self.image_data_output_item = image_data_output_item
        self.image_data_output_measure = image_data_output_measure

        super(AnalysisModel, self).__init__()


class GridModel(model.Model):

    def __init__(self, use_utso=True, median_coefficient=0.99, manual_threshold=0.05, grid=None, gridding_offsets=None):

        self.use_utso = use_utso
        self.median_coefficient = median_coefficient
        self.manual_threshold = manual_threshold
        self.grid = grid
        self.gridding_offsets = gridding_offsets

        super(GridModel, self).__init__()


class XMLModel(model.Model):

    def __init__(self, exclude_compartments=tuple(), exclude_measures=tuple(), make_short_tag_version=True,
                 short_tag_measure=MEASURES.Sum):

        self.exclude_compartments = exclude_compartments
        self.exclude_measures = exclude_measures
        self.make_short_tag_version = make_short_tag_version
        self.short_tag_measure = short_tag_measure

        super(XMLModel, self).__init__()


class AnalysisMetaData(model.Model):

    def __init__(self, start_time=0, name="", description="", interval=20.0, images=0,
                 uuid="", fixture="", scanner="", project_id="", scanner_layout_id="", version=__version__,
                 pinnings=[]):

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