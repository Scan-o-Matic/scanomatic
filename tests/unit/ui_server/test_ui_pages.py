from __future__ import absolute_import
from flask import Flask, template_rendered
import pytest
import os

from scanomatic.ui_server.ui_pages import add_routes
from scanomatic.ui_server import ui_server


@pytest.fixture
def app():
    app = Flask(
        __name__,
        template_folder=os.path.join(
            os.path.dirname(ui_server.__file__), 'templates',
        )
    )
    add_routes(app)
    return app


class TemplateContextProxy(object):
    def __init__(self, app):
        self._context = None
        template_rendered.connect(self._recordcontext, app)

    def _recordcontext(self, sender, context, **extra):
        self._context = context

    def get(self, key, default=None):
        if self._context is None:
            raise RuntimeError('No template has been rendered')
        return self._context.get(key, default)


@pytest.fixture
def templatecontext(app):
    return TemplateContextProxy(app)


class TestCompile:
    def test_no_arguments(self, client, templatecontext):
        client.get('/compile')
        assert templatecontext.get('projectdirectory') == ''
        assert templatecontext.get('projectdirectory_readonly') is False

    def test_given_projectdirectory(self, client, templatecontext):
        client.get('/compile?projectdirectory=foo/bar')
        assert templatecontext.get('projectdirectory') == 'foo/bar'
        assert templatecontext.get('projectdirectory_readonly') is True
