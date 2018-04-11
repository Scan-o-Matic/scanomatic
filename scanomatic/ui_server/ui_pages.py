from __future__ import absolute_import

import os

from flask import (
    send_from_directory, render_template, redirect, abort, request
)

from scanomatic import get_version
from scanomatic.data_processing import phenotyper
from scanomatic.io.paths import Paths
from .general import convert_url_to_path, serve_log_as_html


def add_routes(app):

    @app.route("/")
    def _root():
        return render_template(
            Paths().ui_status_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/home")
    def _show_homescreen():
        return render_template(
            Paths().ui_status_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/ccc")
    def _ccc():
        return render_template(
            Paths().ui_ccc_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/fixtures")
    def _fixtures():
        return render_template(
            Paths().ui_fixture_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/status")
    def _status():
        return render_template(
            Paths().ui_status_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/qc_norm")
    def _qc_norm():
        return render_template(
            Paths().ui_qc_norm_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/help")
    def _help():
        return render_template(
            Paths().ui_help_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/wiki")
    def _wiki():
        return redirect("https://github.com/local-minimum/scanomatic/wiki")

    @app.route("/feature_extract")
    def _feature_extract():
        return render_template(
            Paths().ui_feature_extract_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/analysis")
    def _analysis():
        return render_template(
            Paths().ui_analysis_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/experiment")
    def _experiment():
        return render_template(
            Paths().ui_experiment_file,
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/compile")
    def _compile():
        projectdir = request.args.get('projectdirectory')
        context = {
            'projectdirectory': projectdir if projectdir is not None else '',
            'projectdirectory_readonly': projectdir is not None,
        }
        return render_template(
            Paths().ui_compile_file,
            debug=app.debug,
            version=get_version(),
            **context
        )

    @app.route("/logs/project/<path:project>")
    def _project_logs(project):

        path = convert_url_to_path(project)

        if not os.path.isfile(path):

            abort(404)

        is_project_analysis = phenotyper.path_has_saved_project_state(path)
        include_levels = 3 if is_project_analysis else 2

        return serve_log_as_html(
            path, os.sep.join(path.split(os.path.sep)[-include_levels:]))
