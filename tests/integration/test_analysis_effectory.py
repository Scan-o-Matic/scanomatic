import numpy
import pytest

from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
from scanomatic.server.analysis_effector import AnalysisEffector
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory


@pytest.fixture(scope='session')
def proj1(pytestconfig):
    return pytestconfig.rootdir.join('tests/integration/fixtures/proj1')


def test_colony_sizes(proj1, tmpdir):
    workdir = tmpdir.mkdir('proj1')
    files = [
        'fixture.config',
        'proj1.project.compilation',
        'proj1.project.compilation.instructions',
        'proj1.scan.instructions',
        'proj1_0215_258418.2895.tiff',
    ]
    for filename in files:
        proj1.join(filename).copy(workdir)

    analysis_model = AnalysisModelFactory.create(
        compilation=str(workdir.join('proj1.project.compilation')),
        chain=False,
    )
    assert AnalysisModelFactory.validate(analysis_model)
    job = RPC_Job_Model_Factory.create(id='135', content_model=analysis_model)
    assert RPC_Job_Model_Factory.validate(job)

    analysis_effector = AnalysisEffector(job)
    analysis_effector.setup(job, False)
    for _ in analysis_effector:
        pass

    expected_colony_sizes = numpy.load(str(proj1.join('analysis/image_0_data.npy')))
    actual_colony_sizes = numpy.load(str(workdir.join('analysis/image_0_data.npy')))
    numpy.testing.assert_allclose(expected_colony_sizes, actual_colony_sizes, rtol=.005)
