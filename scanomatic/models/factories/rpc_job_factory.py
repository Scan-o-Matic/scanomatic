import scanomatic.models.rpc_job_models as rpc_job_models
from scanomatic.generics.abstract_model_factory import AbstractModelFactory
from scanomatic.generics.model import Model
from scanomatic.models.factories.scanning_factory import ScanningModel, ScanningModelFactory
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import AnalysisModel
from scanomatic.models.factories.compile_project_factory import CompileProjectFactory
from scanomatic.models.compile_project_model import CompileInstructionsModel
from scanomatic.models.features_model import FeaturesModel
from scanomatic.models.factories.features_factory import FeaturesFactory
from types import StringTypes


class RPC_Job_Model_Factory(AbstractModelFactory):

    MODEL = rpc_job_models.RPCjobModel
    _SUB_FACTORIES = {ScanningModel: ScanningModelFactory,
                      AnalysisModel: AnalysisModelFactory,
                      CompileInstructionsModel: CompileProjectFactory,
                      FeaturesModel: FeaturesFactory
                      }
    STORE_SECTION_HEAD = ('id',)
    STORE_SECTION_SERIALIZERS = {
        'id': str,
        'type': rpc_job_models.JOB_TYPE,
        'status': rpc_job_models.JOB_STATUS,
        'priority': int,
        'content_model': Model,
        'pid': int}

    @classmethod
    def create(cls, **settings):
        """:rtype : scanomatic.models.rpc_job_models.RPCjobModel"""

        return super(RPC_Job_Model_Factory, cls).create(**settings)

    @classmethod
    def _validate_pid(cls, model):

        if model.pid is None or isinstance(model.pid, int) and model.pid > 0:

            return True

        return model.FIELD_TYPES.pid

    @classmethod
    def _validate_id(cls, model):

        if isinstance(model.id, StringTypes):
    
            return True

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
