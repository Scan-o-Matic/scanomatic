import scanomatic.models.rpc_job_models as rpc_job_models
from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.generics.model import Model
from scanomatic.models.factories.scanning_factory import ScanningModel, ScanningModelFactory
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import AnalysisModel

class RPC_Job_Model_Factory(AbstractModelFactory):

    _MODEL = rpc_job_models.RPCjobModel
    _SUB_FACTORIES = {ScanningModel: ScanningModelFactory, AnalysisModel: AnalysisModelFactory }
    STORE_SECTION_HEAD = ('id',)
    STORE_SECTION_SERIALIZERS = {
        ('id',): int,
        ('type',): rpc_job_models.JOB_TYPE,
        ('status',): rpc_job_models.JOB_STATUS,
        ('content_model',): AbstractModelFactory,
        ('pid',): int}

    @classmethod
    def _validate_pid(cls, model):

        if model.pid is None or isinstance(model.pid, int) and model.pid > 0:

            return True

        return model.FIELD_TYPES.pid

    @classmethod
    def _validate_id(cls, model):

        if isinstance(model.id, str):
    
            return True

        #TODO: Add verification of uniqueness?
        return model.FIELD_TYPES.id

    @classmethod
    def _validate_type(cls, model):

        if model.type in rpc_job_models.JOB_TYPE:

            return True

        return model.FIELD_TYPES.type

    @classmethod
    def _validate_priority(cls, model):

        return isinstance(model.priority, int)

    @classmethod
    def _validate_status(cls, model):

        if model.status in rpc_job_models.JOB_STATUS:

            return True

        return model.FIELD_TYPES.model

    @classmethod
    def _vaildate_content_model(cls, model):

        if isinstance(model.content_model, Model):

            return True

        return model.FIELD_TYPES.content_model
