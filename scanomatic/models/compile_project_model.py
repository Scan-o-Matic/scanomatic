__author__ = 'martin'

from enum import Enum
from scanomatic.generics.model import Model

class COMPILE_ACTION(Enum):

    Overwrite = 0
    Append = 1
    Finalize = 2
    FinalizeAndSpawnAnalysis = 3


class CompileInstructionsModel(Model):

    def __init__(self, compile_action=COMPILE_ACTION.Overwrite, images=tuple(), path="", ordinal_number=0):

        super(CompileInstructionsModel, self).__init__(
            compile_action=compile_action,
            images=images,
            path=path,
            ordinal_number=ordinal_number
        )

class CompileImageModel(Model):

    def __init__(self, index=-1, path="", fixture=""):

        super(CompileImage, self).__init__(
            index=index,
            path=path,
            fixture=fixture
        )

