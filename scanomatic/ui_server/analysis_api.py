import os
from flask import Flask, jsonify
from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url, get_search_results
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
    def get_instructions(project=None):

        base_url = "/api/analysis/instructions"

        path = convert_url_to_path(project)

        analysis_file = os.path.join(path, Paths().analysis_model_file)
        model = AnalysisModelFactory.serializer.load_first(analysis_file)
        """:type model: AnalysisModel"""

        if model is None:

            return jsonify(success=True,
                           **get_search_results(path, base_url))

        return jsonify(
            success=True,
            instructions={
                'grayscale': "one-time" if model.one_time_grayscale else "dynamic",
                'positioning': "one-time" if model.one_time_positioning else "dynamic",
                'compilation': model.compilation,
                'compile_instructions': model.compile_instructions,
                'email': model.email,
                'grid_model': {'gridding_offsets': model.grid_model.gridding_offsets,
                               'reference_grid_folder': model.grid_model.reference_grid_folder},
            },
            compile_instructions=convert_path_to_url("/api/compilation/instructions", model.compile_instructions),
            **get_search_results(path, base_url))