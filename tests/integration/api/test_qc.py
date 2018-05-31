from __future__ import absolute_import
import pytest
import mock
import os
import json


@pytest.fixture(scope='session')
def mock_convert_url_to_path():
    m = mock.patch(
        'scanomatic.data_processing.analysis_loader.convert_url_to_path',
        return_value=os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            'fixtures',
            'analysis'
        )),
    )
    yield m.start()
    m.stop()


class TestQCAPI:
    ROUTE = '/qc/growthcurves/{plate}/{path}'

    def test_requesting_growth_curves_for_unknown_project_fails(self, client):
        response = client.get(
            self.ROUTE.format(plate=0, path='missing/project/analysis'),
            follow_redirects=True,
        )
        assert response.status_code == 400

    def test_requesting_growth_curves_for_known_project_succeeds(
        self, mock_convert_url_to_path, client,
    ):
        response = client.get(
            self.ROUTE.format(plate=0, path='project/analysis'),
            follow_redirects=True,
        )
        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['times_data']) == 218
        assert len(data['raw_data']) == 32
        assert len(data['raw_data'][0]) == 48
        assert len(data['raw_data'][0][0]) == 218
        assert len(data['smooth_data']) == 32
        assert len(data['smooth_data'][0]) == 48
        assert len(data['smooth_data'][0][0]) == 218

    def test_requesting_growth_curves_for_unknown_plate_fails(
        self, mock_convert_url_to_path, client,
    ):
        response = client.get(
            self.ROUTE.format(plate=41, path='project/analysis'),
            follow_redirects=True,
        )
        assert response.status_code == 400
