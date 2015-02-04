import scanomatic.generics.model as model
from enum import Enum

JOB_TYPE = Enum("JOB_TYPE",
                 names=("Scan", "Rebuild", "Analysis", "Features", "Unknown")) 

JOB_STATUS = Enum("JOB_STATUS",
                  names=("Requested", "Queued", "Running", "Restoring",
                         "Done", "Unknown"))


class RPCjobModel(model.Model):

    def __init__(self,
                 id=None,
                 type=JOB_TYPE.Unknown,
                 status=JOB_STATUS.Unknown,
                 content_model=None,
                 pid=None):

        super(RPCjobModel, self).__init__(
                id=id, type=type, status=status, pid=pid,
                contentModel=contentModel)
