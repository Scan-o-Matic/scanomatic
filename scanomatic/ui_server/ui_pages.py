from __future__ import absolute_import

import os

from flask import (
    send_from_directory, render_template, redirect, abort, request
)

from scanomatic.data_processing import phenotyper
from scanomatic.io.paths import Paths
from .general import convert_url_to_path, serve_log_as_html


def add_routes(app):

    @app.route("/")
    def _root():
        return render_template(Paths().ui_root_file, debug=app.debug)

    @app.route("/home")
    def _show_homescreen():
        return redirect("/status")

    @app.route("/ccc")
    def _ccc():
        return send_from_directory(Paths().ui_root, Paths().ui_ccc_file)

    @app.route("/fixtures")
    def _fixtures():

        return send_from_directory(Paths().ui_root, Paths().ui_fixture_file)

    @app.route("/status")
    def _status():
        return send_from_directory(Paths().ui_root, Paths().ui_status_file)

    @app.route("/qc_norm")
    def _qc_norm():
        return send_from_directory(Paths().ui_root, Paths().ui_qc_norm_file)

    @app.route("/help")
    def _help():
        return send_from_directory(Paths().ui_root, Paths().ui_help_file)

    @app.route("/wiki")
    def _wiki():
        return redirect("https://github.com/local-minimum/scanomatic/wiki")

    @app.route("/feature_extract", methods=['get'])
    def _feature_extract():

        return send_from_directory(
            Paths().ui_root, Paths().ui_feature_extract_file)

    @app.route("/analysis", methods=['get'])
    def _analysis():

        return send_from_directory(Paths().ui_root, Paths().ui_analysis_file)

    @app.route("/experiment", methods=['get'])
    def _experiment():

        return send_from_directory(Paths().ui_root, Paths().ui_experiment_file)

    @app.route("/compile", methods=['get'])
    def _compile():
        projectdir = request.args.get('projectdirectory')
        context = {
            'projectdirectory': projectdir if projectdir is not None else '',
            'projectdirectory_readonly': projectdir is not None,
        }
        return render_template(Paths().ui_compile_file, **context)

    @app.route("/logs/project/<path:project>")
    def _project_logs(project):

        path = convert_url_to_path(project)

        if not os.path.isfile(path):

            abort(404)

        is_project_analysis = phenotyper.path_has_saved_project_state(path)
        include_levels = 3 if is_project_analysis else 2

        return serve_log_as_html(
            path, os.sep.join(path.split(os.path.sep)[-include_levels:]))
