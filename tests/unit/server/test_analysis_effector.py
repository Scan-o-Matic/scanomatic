from __future__ import absolute_import
import pytest
from mock import patch, MagicMock
import os

from scanomatic.server.analysis_effector import AnalysisEffector
from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
from scanomatic.models.rpc_job_models import JOB_TYPE, JOB_STATUS
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import DEFAULT_PINNING_FORMAT
from tests.factories import make_calibration


@pytest.fixture
def calibrationstore():
    return MagicMock()


@pytest.fixture(autouse=True)
def store_from_env(calibrationstore):
    calibrationstore.get_all_calibrations.return_value = [
        make_calibration(identifier='default', active=True),
    ]
    calibrationstore.get_calibration_by_id.return_value = (
        make_calibration(active=True)
    )
    with patch(
        'scanomatic.models.factories.analysis_factories.store_from_env',
    ) as store_from_env:
        store_from_env.return_value.__enter__.return_value = calibrationstore
        yield


@pytest.fixture
def analysis_job():
    content_model = AnalysisModelFactory.create(
        compilation='fake.project.instructions',
    )

    return RPC_Job_Model_Factory.create(
        id='test',
        type=JOB_TYPE.Analysis,
        status=JOB_STATUS.Running,
        content_model=content_model
    )


@pytest.fixture
def analysis_effector(analysis_job):
    return AnalysisEffector(analysis_job)


@patch.object(AnalysisModelFactory, 'validate', return_value=True)
def test_setup_can_execute(mock, analysis_effector):
    analysis_effector.setup(analysis_effector._job)
    assert analysis_effector._allow_start is True


@patch(
    "scanomatic.io.first_pass_results.CompilationResults",
    return_value=MagicMock(plates=[1, 2, 3]),
)
@patch(
    "scanomatic.image_analysis.analysis_image.ProjectImage",
    return_value=MagicMock(set_grid=lambda: True),
)
@patch.object(AnalysisModelFactory.serializer, "dump")
@patch.object(os, "makedirs")
def test_first_iteration_sets_pinning_formats(
    compilation_results_mock,
    project_image_mock, serializer_dump,
    mkdirs_mock, analysis_effector,
):
    analysis_effector.setup(analysis_effector._job, save_log=False)

    with patch.object(
        analysis_effector, "_remove_files_from_previous_analysis"
    ) as remove_files_mock:

        analysis_effector._setup_first_iteration()

        serializer_dump.assert_called()
        mkdirs_mock.assert_called()
        remove_files_mock.assert_called()
        compilation_results_mock.assert_called()
        project_image_mock.assert_called()

        pinnings = analysis_effector._analysis_job.pinning_matrices
        assert pinnings == [DEFAULT_PINNING_FORMAT] * 3


def test_set_default_pinning_formats(analysis_effector):
    analysis_effector.setup_pinning(13)
    pinnings = analysis_effector._analysis_job.pinning_matrices
    assert pinnings == [DEFAULT_PINNING_FORMAT] * 13
