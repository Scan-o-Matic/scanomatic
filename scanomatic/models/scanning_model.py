import scanomatic.generics.model as model
from scanomatic.generics.enums import MinorMajorStepEnum
from scanomatic.generics.decorators import class_property
import scanomatic
from enum import Enum


class SCAN_CYCLE(MinorMajorStepEnum):

    Unknown = -1
    Wait = 0
    RequestScanner = 10
    WaitForUSB = 11
    ReportNotObtainedUSB = 12
    Scan = 20
    WaitForScanComplete = 21
    ReportScanError = 22
    RequestScannerOff = 30
    VerifyImageSize = 41
    VerifyDiskspace = 42
    RequestProjectCompilation = 50

    @class_property
    def default(cls):
        return cls.Unknown


class SCAN_STEP(Enum):

    Wait = 0
    NextMinor = 1
    NextMajor = 2
    TruncateIteration = 3


class PLATE_STORAGE(Enum):

    Unknown = -1
    Fresh = 0
    Cold = 1
    RoomTemperature = 2


class CULTURE_SOURCE(Enum):

    Unknown = -1
    Freezer80 = 0
    Freezer20 = 1
    Fridge = 2
    Shipped = 3
    Novel = 4


class COMPILE_STATE(Enum):

    NotInitialized = 0
    Initialized = 1
    Finalized = 2


class ScanningAuxInfoModel(model.Model):

    def __init__(
            self,
            stress_level=-1,
            plate_storage=PLATE_STORAGE.Unknown,
            plate_age=-1.0,
            pinning_project_start_delay=-1,
            precultures=-1,
            culture_freshness=-1,
            culture_source=CULTURE_SOURCE.Unknown):

        self.stress_level = stress_level
        self.plate_storage = plate_storage
        self.plate_age = plate_age
        self.pinning_project_start_delay = pinning_project_start_delay
        self.precultures = precultures
        self.culture_freshness = culture_freshness
        self.culture_source = culture_source

        super(ScanningAuxInfoModel, self).__init__()


class ScanningModel(model.Model):

    def __init__(
            self,
            number_of_scans=217,
            time_between_scans=20,
            project_name="",
            directory_containing_project="",
            id="",
            start_time=0.0,
            description="",
            email="",
            pinning_formats=tuple(),
            fixture="",
            scanner=1,
            scanner_hardware="EPSON V700",
            mode="TPU",
            computer="",
            auxillary_info=ScanningAuxInfoModel(),
            plate_descriptions=tuple(),
            version=scanomatic.__version__,
            scanning_program="",
            scanning_program_version="",
            scanning_program_params=tuple(),
            cell_count_calibration_id=None):

        self.number_of_scans = number_of_scans
        self.time_between_scans = time_between_scans
        self.project_name = project_name
        self.directory_containing_project = directory_containing_project
        self.id = id
        self.description = description
        self.plate_descriptions = plate_descriptions
        self.email = email
        """:type : str or [str]"""
        self.pinning_formats = pinning_formats
        self.fixture = fixture
        self.scanner = scanner
        self.scanner_hardware = scanner_hardware
        self.scanning_program = scanning_program
        self.scanning_program_version = scanning_program_version
        self.scanning_program_params = scanning_program_params
        self.mode = mode
        self.computer = computer
        self.start_time = start_time
        self.cell_count_calibration_id = cell_count_calibration_id
        self.auxillary_info = auxillary_info
        self.version = version

        super(ScanningModel, self).__init__()


class PlateDescription(model.Model):

    def __init__(self, name='', index=-1, description=''):

        if name is '':
            name = "Plate {0}".format(index + 1)

        self.name = name
        self.index = index
        self.description = description


class ScannerOwnerModel(model.Model):

    def __init__(self, id=None, pid=0):

        self.id = id
        self.pid = pid
        super(ScannerOwnerModel, self).__init__()


class ScannerModel(model.Model):

    def __init__(
            self,
            socket=-1,
            scanner_name="",
            owner=None, usb="",
            model='',
            power=False,
            last_on=-1,
            last_off=-1,
            expected_interval=0,
            email="", warned=False,
            claiming=False,
            reported=False):

        self.socket = socket
        self.scanner_name = scanner_name
        self.usb = usb
        self.power = power
        self.model = model
        self.last_on = last_on
        self.last_off = last_off
        self.expected_interval = expected_interval
        self.email = email
        self.warned = warned
        self.owner = owner
        self.claiming = claiming
        self.reported = reported

        super(ScannerModel, self).__init__()


class ScanningModelEffectorData(model.Model):

    def __init__(
            self,
            current_cycle_step=SCAN_CYCLE.Wait,
            current_step_start_time=-1,
            current_image=-1,
            current_image_path="",
            current_image_path_pattern="",
            previous_scan_cycle_start=-1.0,
            current_scan_time=-1.0,
            scanner_model='',
            scanning_image_name="",
            usb_port="",
            scanning_thread=None,
            scan_success=False,
            compile_project_model=None,
            known_file_size=0,
            warned_file_size=False,
            warned_scanner_error=False,
            warned_terminated=False,
            warned_scanner_usb=False,
            warned_discspace=False,
            informed_close_to_end=False,
            compilation_state=COMPILE_STATE.NotInitialized):

        self.current_cycle_step = current_cycle_step
        self.current_step_start_time = current_step_start_time
        self.current_image = current_image
        self.current_image_path = current_image_path
        self.current_image_path_pattern = current_image_path_pattern
        self.previous_scan_cycle_start = previous_scan_cycle_start
        self.current_scan_time = current_scan_time
        self.scanning_thread = scanning_thread
        self.scan_success = scan_success
        self.scanning_image_name = scanning_image_name
        self.usb_port = usb_port
        self.scanner_model = scanner_model
        self.compile_project_model = compile_project_model
        """:type : scanomatic.models.compile_project_model.CompileInstructionsModel"""
        self.known_file_size = known_file_size
        self.warned_file_size = warned_file_size
        self.warned_scanner_error = warned_scanner_error
        self.warned_scanner_usb = warned_scanner_usb
        self.warned_discspace = warned_discspace
        self.warned_terminated = warned_terminated
        self.compilation_state = compilation_state
        self.informed_close_to_end = informed_close_to_end

        super(ScanningModelEffectorData, self).__init__()
