__version__ = "0.9991"

import scanomatic.generics.model as model
from scanomatic.generics.enums import MinorMajorStepEnum
from enum import Enum


class SCAN_CYCLE(MinorMajorStepEnum):

    Wait = 0
    RequestScanner = 10
    WaitForUSB = 11
    ReportNotObtainedUSB = 12
    Scan = 20
    WaitForScanComplete = 21
    ReportScanError = 22
    RequestScannerOff = 30
    RequestFirstPassAnalysis = 40


class SCAN_STEP(Enum):

    Wait = 0
    NextMinor = 1
    NextMajor = 2


class ScanningModel(model.Model):

    def __init__(self, number_of_scans=217, time_between_scans=20,
                 project_name="", directory_containing_project="",
                 project_tag="", scanner_tag="", id="",
                 description="", email="", pinning_formats=tuple(),
                 fixture="", scanner=1, scanner_hardware="EPSON V700", mode="TPU",  version=__version__):

        super(ScanningModel, self).__init__(
            number_of_scans=number_of_scans, time_between_scans=time_between_scans,
            project_name=project_name,
            directory_containing_project=directory_containing_project,
            project_tag=project_tag, scanner_tag=scanner_tag, id=id,
            description=description, email=email,
            pinning_formats=pinning_formats,
            fixture=fixture, scanner=scanner, scanner_hardware=scanner_hardware, mode=mode, version=version)


class ScannerOwnerModel(model.Model):

    def __init__(self, socket=-1, scanner_name="", owner=None, usb="", power=False, last_on=-1, last_off=-1,
                 expected_interval=0, email="", warned=False, claiming=False, reported=False):

        super(ScannerOwnerModel, self).__init__(socket=socket, scanner_name=scanner_name, usb=usb,
                                                power=power, last_on=last_on, last_off=last_off,
                                                expected_interval=expected_interval, email=email, warned=warned,
                                                owner=owner, claiming=claiming, reported=reported)


class ScanningModelEffectorData(model.Model):

    def __init__(self, current_cycle_step=SCAN_CYCLE.Wait, current_step_start_time=-1, current_image=-1,
                 current_image_path="", current_image_path_pattern="",
                 project_time=-1.0, previous_scan_time=-1.0, images_ready_for_first_pass_analysis=[],
                 scanning_image_name="", usb_port="", scanning_thread=None, scan_success=False,
                 previous_compile_job=None):

        super(ScanningModelEffectorData, self).__init__(
            current_cycle_step=current_cycle_step,
            current_step_start_time=current_step_start_time,
            current_image=current_image,
            current_image_path=current_image_path,
            current_image_path_pattern=current_image_path_pattern,
            project_time=project_time,
            previous_scan_time=previous_scan_time,
            scanning_thread=scanning_thread,
            scan_success=scan_success,
            images_ready_for_first_pass_analysis=images_ready_for_first_pass_analysis,
            scanning_image_name=scanning_image_name,
            usb_port=usb_port,
            previous_compile_job=previous_compile_job)