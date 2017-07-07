import os
from itertools import chain, product
from glob import glob
from flask import jsonify, request
from scanomatic.ui_server.general import (
    convert_url_to_path, convert_path_to_url, get_search_results, json_response
)
from scanomatic.io.paths import Paths
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.analysis_model import DefaultPinningFormats
from scanomatic.image_analysis.grid_array import GridArray
from .general import get_image_data_as_array, json_abort


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/analysis/pinning/formats")
    def get_supported_pinning_formats():

        return jsonify(
            is_endpoint=True,
            pinning_formats=[
                dict(
                    name=pinning.human_readable(),
                    value=pinning.value,
                )
                for pinning in DefaultPinningFormats
            ]
        )

    @app.route("/api/analysis/image/grid", methods=['POST'])
    def get_gridding_image():

        pinning_format = request.values.get_list('pinning_format')
        correction = request.values.getlist('gridding_correction')
        if not correction:
            correction = None
        image = get_image_data_as_array(request.files.get('image'))

        analysis_model = AnalysisModelFactory.create()
        analysis_model.output_directory = ""
        grid_array = GridArray((None, None), pinning_format, analysis_model)

        if not grid_array.detect_grid(image, grid_correction=correction):
            return json_abort(
                400,
                reason="Grid detection failed",
                is_endpoint=True,
            )

        grid = grid_array.grid
        inner = len(grid[0])
        outer = len(grid)
        xy1 = [([None] for _ in range(inner)) for _ in range(outer)]
        xy2 = [([None] for _ in range(inner)) for _ in range(outer)]

        for pos in product(range(outer), range(inner)):

            outr, innr = pos
            grid_cell = grid_array[pos]
            xy1[outr][innr] = grid_cell.xy1
            xy2[outr][innr] = grid_cell.xy2

        return jsonify(
            is_endpoint=True,
            xy1=xy1,
            xy2=xy2,
            grid=grid
        )

    @app.route("/api/analysis/instructions", defaults={'project': ''})
    @app.route("/api/analysis/instructions/", defaults={'project': ''})
    @app.route("/api/analysis/instructions/<path:project>")
    def get_analysis_instructions(project=None):

        base_url = "/api/analysis/instructions"

        path = convert_url_to_path(project)

        analysis_file = os.path.join(path, Paths().analysis_model_file)
        model = AnalysisModelFactory.serializer.load_first(analysis_file)
        """:type model: scanomatic.models.analysis_model.AnalysisModel"""

        analysis_logs = tuple(chain(((
            convert_path_to_url("/api/tools/logs/0/0", c),
            convert_path_to_url(
                "/api/tools/logs/WARNING_ERROR_CRITICAL/0/0", c)) for c in
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
                    'grayscale': "one-time" if model.one_time_grayscale
                    else "dynamic",
                    'positioning': "one-time" if model.one_time_positioning
                    else "dynamic",
                    'compilation': model.compilation,
                    'compile_instructions': model.compile_instructions,
                    'email': model.email,
                    'pinning_matrices': model.pinning_matrices,
                    'grid_model': {
                        'gridding_offsets': model.grid_model.gridding_offsets,
                        'reference_grid_folder':
                        model.grid_model.reference_grid_folder
                    },
                },
                analysis_logs=analysis_logs,
                compile_instructions=[convert_path_to_url(
                    "/api/compile/instructions", model.compile_instructions)],
                **get_search_results(path, base_url))))
