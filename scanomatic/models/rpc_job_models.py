import scanomatic.generics.model as model
import scanomatic.generics.decorators as decorators
from scanomatic.generics.enums import CycleEnum
from enum import Enum
import numpy as np


class JOB_TYPE(CycleEnum):

    Scan = ("Scanner Job", 0)
    Compile = ("Compile Project", 1)
    Analysis = ("Analysis Job", 10)
    Features = ("Feature Extraction Job", 20)
    Unknown = ("Unknown Job", -1)

    @property
    def int_value(self):
        return self.value[1]

    @property
    def text(self):
        return self.value[0]

    @property
    def next(self):

        next_int_val = (1 + np.round(self.int_value/10.)) * 10
        return self.get_by_int_representation(next_int_val)

    @classmethod
    def get_by_int_representation(cls, value):
        for member in cls.__members__.values():
            if value == member.int_value:
                return member
        return cls.default

    @decorators.class_property
    def default(cls):
        return cls.Unknown


JOB_STATUS = Enum("JOB_STATUS",
                  names=("Requested", "Queued", "Running", "Restoring",
                         "Done", "Aborted", "Crashed", "Unknown"))


class RPCjobModel(model.Model):

    def __init__(self,
                 id=None,
                 type=JOB_TYPE.Unknown,
                 status=JOB_STATUS.Unknown,
                 content_model=None,
                 priority=-1,
                 pid=None):

        self.id = id
        self.type = type
        self.status = status
        self.pid = pid
        self.priority = priority
        self.content_model = content_model

        super(RPCjobModel, self).__init__()

    def __hash__(self):

        return hash(self.id)