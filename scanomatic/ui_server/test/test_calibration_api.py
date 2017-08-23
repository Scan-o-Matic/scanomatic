import pytest
from flask import Flask

from scanomatic.ui_server import calibration_api
from scanomatic.io.paths import Paths


@pytest.fixture(scope="function")
def test_app():
    app = Flask("Scan-o-Matic UI", template_folder=Paths().ui_templates)
    calibration_api.add_routes(app)
    app.testing = True
    return app.test_client()


class TestFinalizeEndpoint:
    route = "/api/data/calibration/{identifier}/finalize"

    def test_token_not_valid(self, test_app):
        response = test_app.post(
            self.route.format(identifier="test"),
            data={},
            follow_redirects=True)
        assert '401' in response.status
