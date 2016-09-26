import os
from itertools import chain
from glob import glob
from flask import Flask, jsonify
from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url, get_search_results, json_response
from scanomatic.io.paths import Paths
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import AnalysisModel


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/analysis/instructions", defaults={'project': ''})
    @app.route("/api/analysis/instructions/", defaults={'project': ''})
    @app.route("/api/analysis/instructions/<path:project>")
    def get_analysis_instructions(project=None):

        base_url = "/api/analysis/instructions"

        path = convert_url_to_path(project)

        analysis_file = os.path.join(path, Paths().analysis_model_file)
        model = AnalysisModelFactory.serializer.load_first(analysis_file)
        """:type model: AnalysisModel"""

        analysis_logs = tuple(chain(((
            convert_path_to_url("/api/tools/logs/0/0", c),
            convert_path_to_url("/api/tools/logs/WARNING_ERROR_CRITICAL/0/0", c)) for c in
            glob(os.path.join(path, Paths().analysis_run_log)))))

        if model is None:

            return jsonify(**json_response(
                ["urls", "analysis_logs"],
                dict(
                    analysis_logs=analysis_logs,
                    **get_search_results(path, base_url))))

        return jsonify(**json_response(
            ["urls", "compile_instructions", "analysis_logs"],
            dict(
                instructions={
                    'grayscale': "one-time" if model.one_time_grayscale else "dynamic",
                    'positioning': "one-time" if model.one_time_positioning else "dynamic",
                    'compilation': model.compilation,
                    'compile_instructions': model.compile_instructions,
                    'email': model.email,
                    'grid_model': {'gridding_offsets': model.grid_model.gridding_offsets,
                                   'reference_grid_folder': model.grid_model.reference_grid_folder},
                },
                analysis_logs=analysis_logs,
                compile_instructions=[convert_path_to_url("/api/compile/instructions", model.compile_instructions)],
                **get_search_results(path, base_url))))
