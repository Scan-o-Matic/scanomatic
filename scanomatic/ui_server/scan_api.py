from __future__ import absolute_import

from glob import glob
from itertools import chain
import os

from flask import jsonify

from scanomatic.io.paths import Paths
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
from scanomatic.ui_server.general import (
    convert_path_to_url, convert_url_to_path, get_search_results,
    json_response
)


def add_routes(app):
    """

    :param app: The flask webb app
     :type app: Flask
    :return:
    """

    @app.route("/api/scan/instructions", defaults={'project': ''})
    @app.route("/api/scan/instructions/", defaults={'project': ''})
    @app.route("/api/scan/instructions/<path:project>")
    def get_scan_instructions(project=None):

        base_url = "/api/scan/instructions"

        path = convert_url_to_path(project)

        model = ScanningModelFactory.serializer.load_first(path)
        """:type model: scanomatic.models.scanning_model.ScanningModel"""

        if model is None:
            compile_instructions = [
                convert_path_to_url("/api/compile/instructions", p) for p in
                glob(os.path.join(
                    path,
                    Paths().project_compilation_instructions_pattern.format(
                        "*")))
            ]

        else:
            compile_instructions = [
                convert_path_to_url("/api/compile/instructions", p) for p in
                glob(os.path.join(
                    os.path.dirname(path),
                    Paths().project_compilation_instructions_pattern.format(
                        "*")))
            ]

        scan_instructions = [
            convert_path_to_url(base_url, c) for c in
            glob(os.path.join(
                path, Paths().scan_project_file_pattern.format("*")))]

        scan_logs = tuple(chain(((
            convert_path_to_url("/api/tools/logs/0/0", c),
            convert_path_to_url(
                "/api/tools/logs/WARNING_ERROR_CRITICAL/0/0", c)) for c in
                glob(os.path.join(
                    path, Paths().scan_log_file_pattern.format("*"))))))

        if model is not None:

            return jsonify(**json_response(
                [
                    "urls",
                    "scan_instructions",
                    "compile_instructions",
                    "scan_logs"
                ],
                dict(
                    instructions={
                        'fixture': model.fixture,
                        'computer': model.computer,
                        'description': model.description,
                        'number_of_scans': model.number_of_scans,
                        'mode': model.mode,
                        'plate_descriptions': model.plate_descriptions,
                        'pinning_formats': model.pinning_formats,
                        'time_between_scans': model.time_between_scans,
                        'start_time': model.start_time,
                        'project_name': model.project_name,
                        'scanner': model.scanner,
                        'scanner_hardware': model.scanner_hardware,
                        'auxillary_info': {
                            'culture_freshness':
                            model.auxillary_info.culture_freshness if
                            model.auxillary_info.culture_freshness else None,
                            'culture_source':
                            model.auxillary_info.culture_source.name if
                            model.auxillary_info.culture_source else None,
                            'pinning_project_start_delay':
                            model.auxillary_info.pinning_project_start_delay,
                            'plate_age': model.auxillary_info.plate_age,
                            'precultures': model.auxillary_info.precultures,
                            'plate_storage':
                            model.auxillary_info.plate_storage.name if
                            model.auxillary_info.plate_storage else None,
                            'stress_level': model.auxillary_info.stress_level,
                        },
                        'email': model.email,
                    },
                    compile_instructions=compile_instructions,
                    scan_instructions=scan_instructions,
                    scan_logs=scan_logs,
                    **get_search_results(path, base_url))))
        else:
            return jsonify(**json_response(
                [
                    "urls",
                    "scan_instructions",
                    "compile_instructions",
                    "scan_logs"
                ],
                dict(
                    compile_instructions=compile_instructions,
                    scan_instructions=scan_instructions,
                    scan_logs=scan_logs,
                    **get_search_results(path, base_url))))
