import pytest

from flask import Flask

from scanomatic.ui_server import general


@pytest.fixture(scope='module')
def app():

    _app = Flask("--dummy--")
    with _app.app_context():
        yield _app


class TestJsonAbort:
    @pytest.mark.parametrize(
        "status_code,args,kwargs",
        [
            (300, [], {}),
            (400, [1, 2], {}),
            (500, [], {'3': 1, '1': 20}),
        ]
    )
    def test_json_abort(self, app, status_code, args, kwargs):

        with app.test_request_context():
            assert general.json_abort(
                status_code, *args, **kwargs).status_code == status_code

    def test_json_abort_raises(self, app):
        with app.test_request_context():
            with pytest.raises(TypeError):
                assert general.json_abort(600, *[42], **{'20': 21, '21': 20})
