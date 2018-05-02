from __future__ import absolute_import

import os

from flask import abort, redirect, render_template, request

from scanomatic import get_version
from scanomatic.data_processing import phenotyper
from .general import convert_url_to_path, serve_log_as_html


def add_routes(app):

    @app.route("/")
    def _root():
        return render_template(
            'status.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/home")
    def _show_homescreen():
        return render_template(
            'status.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/ccc")
    def _ccc():
        return render_template(
            'CCC.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/fixtures")
    def _fixtures():
        return render_template(
            'fixture.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/status")
    def _status():
        return render_template(
            'status.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/qc_norm")
    def _qc_norm():
        return render_template(
            'qc_norm.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/help")
    def _help():
        return render_template(
            'help.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/wiki")
    def _wiki():
        return redirect("https://github.com/local-minimum/scanomatic/wiki")

    @app.route("/feature_extract")
    def _feature_extract():
        return render_template(
            'feature_extract.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/analysis")
    def _analysis():
        compilation_file = request.args.get('compilationfile')
        context = {
            'compilation_file':
                compilation_file if compilation_file is not None else '',
            'compilation_file_readonly': compilation_file is not None,
        }
        return render_template(
            'analysis.html',
            debug=app.debug,
            version=get_version(),
            **context
        )

    @app.route("/experiment")
    def _experiment():
        return render_template(
            'experiment.html',
            debug=app.debug,
            version=get_version(),
        )

    @app.route("/projects")
    def _projects():
        return render_template(
            'projects.html',
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
            'compile.html',
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
