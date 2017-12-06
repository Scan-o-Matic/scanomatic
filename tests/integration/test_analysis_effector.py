from __future__ import absolute_import
from collections import namedtuple

import numpy
import pytest

from scanomatic.models.factories.rpc_job_factory import RPC_Job_Model_Factory
from scanomatic.server.analysis_effector import AnalysisEffector
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory


@pytest.fixture(scope='session')
def proj1(pytestconfig):
    return pytestconfig.rootdir.join('tests/integration/fixtures/proj1')


@pytest.fixture
def proj1_analysis(proj1, tmpdir):
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
    return namedtuple('proj1_analysis', 'job, workdir')(job, workdir)


def test_colony_sizes(proj1, proj1_analysis):
    analysis_effector = AnalysisEffector(proj1_analysis.job)
    analysis_effector.setup(proj1_analysis.job, False)
    for _ in analysis_effector:
        pass

    expected = numpy.load(str(proj1.join('analysis/image_0_data.npy')))
    actual = numpy.load(
        str(proj1_analysis.workdir.join('analysis/image_0_data.npy')))
    numpy.testing.assert_allclose(expected, actual, rtol=.01)


def test_grid_plate(proj1, proj1_analysis):
    analysis_effector = AnalysisEffector(proj1_analysis.job)
    analysis_effector.setup(proj1_analysis.job, False)
    for _ in analysis_effector:
        pass

    expected = numpy.load(str(proj1.join('analysis/grid_plate___1.npy')))
    actual = numpy.load(
        str(proj1_analysis.workdir.join('analysis/grid_plate___1.npy')))
    numpy.testing.assert_allclose(expected, actual, atol=3)


def test_grid_size(proj1, proj1_analysis):
    analysis_effector = AnalysisEffector(proj1_analysis.job)
    analysis_effector.setup(proj1_analysis.job, False)
    for _ in analysis_effector:
        pass

    expected = numpy.load(str(proj1.join('analysis/grid_size___1.npy')))
    actual = numpy.load(
        str(proj1_analysis.workdir.join('analysis/grid_size___1.npy')))
    assert (expected == actual).all()
