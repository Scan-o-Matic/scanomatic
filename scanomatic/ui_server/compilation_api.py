import os
from flask import Flask, jsonify
from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url, get_search_results
from scanomatic.io.paths import Paths
from glob import glob
from scanomatic.models.compile_project_model import CompileInstructionsModel
from scanomatic.models.factories.compile_project_factory import CompileProjectFactory


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/compile/instructions", defaults={'project': ''})
    @app.route("/api/compile/instructions/", defaults={'project': ''})
    @app.route("/api/compile/instructions/<path:project>")
    def get_compile_instructions(project=None):

        base_url = "/api/compile/instructions"

        path = convert_url_to_path(project)

        model = CompileProjectFactory.serializer.load_first(path)
        """:type model: CompileInstructionsModel"""

        if model is None:
            scan_instructions = [convert_path_to_url("/api/scan/instructions", p) for p in
                                 glob(os.path.join(path, Paths().scan_project_file_pattern.format("*")))]

        else:
            scan_instructions = [convert_path_to_url("/api/scan/instructions", p) for p in
                                 glob(os.path.join(os.path.dirname(path),
                                                   Paths().scan_project_file_pattern.format("*")))]

        compile_instructions = [convert_path_to_url(base_url, c) for c in
                                glob(os.path.join(path, Paths().project_compilation_instructions_pattern.format("*")))]

        if model is not None:

            return jsonify(
                success=True,
                instructions={
                    'fixture': model.fixture_name,
                    'fixture_type': model.fixture_type.name,
                    'compilation': [dict(**i) for i in model.images],
                    'email': model.email,
                },
                compile_instructions=compile_instructions if compile_instructions else None,
                scan_instructions=scan_instructions if scan_instructions else None,
                **get_search_results(path, base_url))

        else:
            return jsonify(
                compile_instructions=compile_instructions if compile_instructions else None,
                scan_instructions=scan_instructions if scan_instructions else None,
                success=True,
                **get_search_results(path, base_url))
