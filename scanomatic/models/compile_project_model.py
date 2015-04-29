__author__ = 'martin'

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

    def __init__(self, compile_action=COMPILE_ACTION.InitiateAndSpawnAnalysis, images=tuple(), path="",
                 start_condition="", fixture=FIXTURE.Local, fixture_name=None):

        super(CompileInstructionsModel, self).__init__(
            compile_action=compile_action,
            images=images,
            path=path,
            start_condition=start_condition,
            fixture=fixture,
            fixture_name=fixture_name
        )


class CompileImageModel(Model):

    def __init__(self, index=-1, path="", time_stamp=0.0):

        super(CompileImageModel, self).__init__(
            index=index,
            path=path,
            time_stamp=time_stamp
        )