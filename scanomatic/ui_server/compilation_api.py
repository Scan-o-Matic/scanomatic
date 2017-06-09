import os
from itertools import chain
from glob import glob

from flask import Flask, jsonify

from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url, get_search_results, json_response, \
    serve_numpy_as_image

from scanomatic.io.paths import Paths
from scanomatic.models.factories.compile_project_factory import CompileProjectFactory
from scanomatic.io import image_loading
from scanomatic.data_processing import phenotyper


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/compile/colony_image")
    @app.route("/api/compile/colony_image/")
    @app.route("/api/compile/colony_image/<int:time_index>/<int:plate>/<int:outer>/<int:inner>/<path:project>")
    @app.route("/api/compile/colony_image/<int:plate>/<int:outer>/<int:inner>/<path:project>")
    def get_colony_image(time_index=0, plate=None, outer=None, inner=None, project=None):
        base_url = "/api/compile/colony_image"

        path = convert_url_to_path(project)

        is_project = phenotyper.path_has_saved_project_state(path)

        if not is_project:
            return jsonify(success=True, is_project=False, is_endpoint=False,
                           **get_search_results(path, base_url))

        im = image_loading.load_colony_image((plate, outer, inner), analysis_directory=path, time_index=time_index)

        return serve_numpy_as_image(im)

    @app.route("/api/compile/instructions", defaults={'project': ''})
    @app.route("/api/compile/instructions/", defaults={'project': ''})
    @app.route("/api/compile/instructions/<path:project>")
    def get_compile_instructions(project=None):

        base_url = "/api/compile/instructions"

        path = convert_url_to_path(project)

        model = CompileProjectFactory.serializer.load_first(path)
        """:type model: scanomatic.models.compile_project_model.CompileInstructionsModel"""

        if model is None:
            scan_instructions = [convert_path_to_url("/api/scan/instructions", p) for p in
                                 glob(os.path.join(path, Paths().scan_project_file_pattern.format("*")))]
        else:
            scan_instructions = [convert_path_to_url("/api/scan/instructions", p) for p in
                                 glob(os.path.join(os.path.dirname(path),
                                                   Paths().scan_project_file_pattern.format("*")))]

        compile_instructions = [convert_path_to_url(base_url, c) for c in
                                glob(os.path.join(path, Paths().project_compilation_instructions_pattern.format("*")))]

        compile_logs = tuple(chain(((
            convert_path_to_url("/api/tools/logs/0/0", c),
            convert_path_to_url("/api/tools/logs/WARNING_ERROR_CRITICAL/0/0", c)) for c in
            glob(os.path.join(path, Paths().project_compilation_log_pattern.format("*"))))))

        if model is not None:

            return jsonify(**json_response(
                ["urls", "compile_instructions", "scan_instructions", "compile_logs"],
                dict(
                    instructions={
                        'fixture': model.fixture_name,
                        'fixture_type': model.fixture_type.name,
                        'compilation': [dict(**i) for i in model.images],
                        'email': model.email,
                        'pinning_matrices': model.overwrite_pinning_matrices,
                        'start_time': model.start_time,
                    },
                    compile_instructions=compile_instructions,
                    scan_instructions=scan_instructions,
                    compile_logs=compile_logs,
                    **get_search_results(path, base_url))))

        else:
            return jsonify(**json_response(
                ["urls", "compile_instructions", "scan_instructions", "compile_logs"],
                dict(
                    compile_instructions=compile_instructions,
                    scan_instructions=scan_instructions,
                    compile_logs=compile_logs,
                    **get_search_results(path, base_url))))

    @app.route("/api/compile/image_list/<location>/<path:project>")
    def get_compile_image_list(location='root', project=None):

        if location != 'root':
            return jsonify(success=False, reason='Illegal location')

        path = convert_url_to_path(project)
        model = CompileProjectFactory.dict_from_path_and_fixture(path)
        return jsonify(images=[{"index": m['index'], 'file': os.path.basename(m['path'])} for m in model['images']])
