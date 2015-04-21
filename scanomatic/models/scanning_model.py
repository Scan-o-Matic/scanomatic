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
                 project_tag="", scanner_tag="",
                 description="", email="", pinning_formats=tuple(),
                 fixture="", scanner=1, mode="TPU", version=__version__):

        super(ScanningModel, self).__init__(
            number_of_scans=number_of_scans, time_between_scans=time_between_scans,
            project_name=project_name,
            directory_containing_project=directory_containing_project,
            project_tag=project_tag, scanner_tag=scanner_tag,
            description=description, email=email,
            pinning_formats=pinning_formats,
            fixture=fixture, scanner=scanner, mode=mode, version=__version__)

class ScannerOwnerModel(model.Model):

    def __init__(self, socket=-1, scanner_name="", job_id="", usb="", power=False, last_on=-1, last_off=-1,
                 expected_interval=0, email="", warned=False, owner_pid=-1, claiming=False):

        super(ScannerOwnerModel, self).__init__(socket=socket, scanner_name=scanner_name, job_id=job_id, usb=usb,
                                                power=power, last_on=last_on, last_off=last_off,
                                                expected_interval=expected_interval, email=email, warned=warned,
                                                owner_pid=owner_pid, claiming=claiming)