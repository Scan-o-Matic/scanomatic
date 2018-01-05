from flask import request, Flask, jsonify
from itertools import product, chain
import os
import glob
from urllib import unquote

from scanomatic.ui_server.general import safe_directory_name
from scanomatic.io.app_config import Config
from scanomatic.io.logger import Logger, parse_log_file
from scanomatic.io.paths import Paths
from scanomatic.data_processing.phenotyper import path_has_saved_project_state
from .general import convert_url_to_path, json_response, serve_zip_file

_logger = Logger("Tools API")


def valid_range(settings):
    return 'min' in settings and 'max' in settings


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/tools/system_logs")
    @app.route("/api/tools/system_logs/<what>/<detail>")
    @app.route("/api/tools/system_logs/<what>")
    def system_log_view(what=None, detail=None):

        base_url = "/api/tools/system_logs"
        if what == 'server':
            what = Paths().log_server
        elif what == "ui_server":
            what = Paths().log_ui_server
        else:

            return jsonify(**json_response(
                ['urls'],
                dict(
                    urls=[
                        "{0}/{1}".format(base_url, w) for w in
                        ('server', 'ui_server')
                    ],
                )
            ))

        try:
            data = parse_log_file(what)
        except IOError:
            return jsonify(success=False, is_endpoint=True, reason="No log-file found with that name")

        return jsonify(success=True, is_endpoint=True, **{k: v for k, v in data.iteritems() if k not in ('file',)})

    @app.route("/api/tools/logs")
    @app.route("/api/tools/logs/<filter_status>/<path:project>")
    @app.route("/api/tools/logs/<int:n_records>/<path:project>")
    @app.route("/api/tools/logs/<filter_status>/<int:n_records>/<path:project>")
    @app.route("/api/tools/logs/<int:start_at>/<int:n_records>/<path:project>")
    @app.route("/api/tools/logs/<filter_status>/<int:start_at>/<int:n_records>/<path:project>")
    def log_view(project='', filter_status=None, n_records=-1, start_at=0):

        # base_url = "/api/tools/logs"
        path = convert_url_to_path(project)
        if n_records == 0:
            n_records = -1

        try:
            data = parse_log_file(path, seek=start_at, max_records=n_records, filter_status=filter_status)
        except IOError:
            return jsonify(success=False, is_endpoint=True, reason="No log-file found with that name")

        return jsonify(success=True, is_endpoint=True, **{k: v for k, v in data.iteritems() if k not in ('file',)})

    @app.route("/api/tools/selection", methods=['POST'])
    @app.route("/api/tools/selection/<operation>", methods=['POST'])
    def tools_create_selection(operation='rect'):
        """Converts selection ranges to api-understood selections.

        _Note_ that the range-boundary uses inclusive min and
        exclusive max indices.

        Query should be json-formatted and each key to be used must
        have the following structure

        ```key: {min: min_value, max: max_value}```

        Args:
            operation: Optional, 'rect' is default.
                Defines how the selection should be made
                * if 'rect' then supplied json-key ranges are treated as
                    combined bounding box coordinates
                * if 'separate' each json-key range is converted
                    individually.


        Returns: json-object containing the same keys as the object sent.

        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if data_object is None or len(data_object) == 0:
            return jsonify(success=False, reason="No valid json or post is empty")

        if operation == 'separate':
            response = {}
            for key in data_object:

                settings = data_object.get(key)

                if valid_range(settings):
                    response[key] = range(settings['min'], settings['max'])

            return jsonify(**response)

        elif operation == 'rect':
            return jsonify(
                **{k: v for k, v in
                   zip(data_object,
                       zip(*product(*(range(v['min'], v['max'])
                                      for k, v in data_object.iteritems()
                                      if valid_range(v)))))})

        else:
            return jsonify()

    @app.route("/api/tools/coordinates", methods=['POST'])
    @app.route("/api/tools/coordinates/<operation>", methods=['POST'])
    def tools_coordinates(operation='create'):
        """Conversion between coordinates and api selections.

        Coordinates are (x, y) positions.

        Selections are separate arrays of X and Y that combined
        makes coordinates.

        Args:
            operation: Optional, default 'create'
                * if 'create' uses the keys supplied in 'keys' or all
                    keys POSTed to create coordinates.
                * if 'parse' converts a list of coordinates under the
                    POSTed key 'coordinates' to selection structure.
        Returns: json-object

        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if operation == 'create':
            keys = data_object.get('keys', data_object.keys())
            return jsonify(coordinates=zip(*(data_object[k] for k in keys)))

        elif operation == 'parse':
            _logger.info("Parsing {0}".format(data_object))
            if 'coordinates' in data_object:
                return jsonify(selection=zip(*data_object['coordinates']))
            else:
                return jsonify(success=False, reason="No coordinates in {0}".format(data_object))

    @app.route("/api/tools/debug/data/<path:project>")
    def _debug_data(project=""):
        """Creates zip of relevant files for a project.

        Args:
            project: Path to the project

        Optional Json/POST/GET data:

            include_global_logs:

                0 = No
                1 = Include active server & ui_server logs
                Note that this will cause zip to be informative of
                general directory structure.

            include_state:
                0 = No
                1 = Include files from feature extraction
                Only valid if pointing at analysis folder

        Returns: Zip-file or json on fail
        """
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        path = convert_url_to_path(project)
        files = []

        if path_has_saved_project_state(path):

            include_state = bool(data_object.get("include_state", default=False))
            proj_path = os.path.abspath(os.path.join(path, os.path.pardir))

            files += glob.glob(os.path.join(path, Paths().analysis_run_log))
            files += glob.glob(os.path.join(path, Paths().analysis_model_file))
            files += glob.glob(os.path.join(path, Paths().phenotypes_extraction_instructions))
            files += glob.glob(os.path.join(path, Paths().phenotypes_extraction_log))

            if include_state:

                files += glob.glob(os.path.join(path, Paths().phenotype_times))
                files += glob.glob(os.path.join(path, Paths().phenotypes_extraction_params))
                files += glob.glob(os.path.join(path, Paths().phenotypes_filter))
                files += glob.glob(os.path.join(path, Paths().phenotypes_filter_undo))
                files += glob.glob(os.path.join(path, Paths().phenotypes_input_data))
                files += glob.glob(os.path.join(path, Paths().phenotypes_input_smooth))
                files += glob.glob(os.path.join(path, Paths().phenotypes_meta_data))
                files += glob.glob(os.path.join(path, Paths().phenotypes_meta_data_original_file_patern))
                files += glob.glob(os.path.join(path, Paths().vector_meta_phenotypes_raw))
                files += glob.glob(os.path.join(path, Paths().vector_phenotypes_raw))
                files += glob.glob(os.path.join(path, Paths().phenotypes_reference_offsets))
                files += glob.glob(os.path.join(path, Paths().experiment_grid_image_pattern.format("*")))

        elif glob.glob(os.path.join(path,
                                    Paths().experiment_scan_image_pattern.format("*", "*", 0).replace("0", "*"))):

            proj_path = path

        else:

            return jsonify(success=False, is_endpoint=True, reason="Not a project")

        files += glob.glob(os.path.join(proj_path, Paths().experiment_local_fixturename))
        files += glob.glob(os.path.join(proj_path, Paths().scan_log_file_pattern.format("*")))
        files += glob.glob(os.path.join(proj_path, Paths().project_compilation_log_pattern.format("*")))
        files += glob.glob(os.path.join(proj_path, Paths().project_compilation_pattern.format("*")))
        files += glob.glob(os.path.join(proj_path, Paths().project_compilation_from_scanning_pattern.format("*")))
        files += glob.glob(os.path.join(proj_path, Paths().project_compilation_from_scanning_pattern_old.format("*")))
        files += glob.glob(os.path.join(proj_path, Paths().project_compilation_instructions_pattern.format("*")))

        if bool(data_object.get("include_global_logs", default=False)):

            files += glob.glob(Paths().log_server)
            files += glob.glob(Paths().log_ui_server)

        return serve_zip_file("DebugFiles_{0}.zip".format(os.path.basename(proj_path)), *files)

    @app.route("/api/tools/path")
    @app.route("/api/tools/path/<command>", methods=['get', 'post'])
    @app.route("/api/tools/path/<command>/", methods=['get', 'post'])
    @app.route("/api/tools/path/<command>/<path:sub_path>", methods=['get', 'post'])
    def _experiment_commands(command=None, sub_path=""):

        if command is None:
            command = 'root'

        sub_path = unquote(sub_path).split("/")

        try:
            is_directory = bool(request.values.get('isDirectory', type=int, default=True))
        except ValueError:
            is_directory = True
        try:
            check_has_analysis = bool(request.values.get('checkHasAnalysis', type=int, default=False))
        except ValueError:
            check_has_analysis = False

        if not all(safe_directory_name(name) for name in sub_path[:None if is_directory else -1]):

            return jsonify(path=Config().paths.projects_root, valid_parent=False,
                           reason="Only letter, numbers and underscore allowed")

        if command == 'root':

            suffix = request.values.get('suffix', default="")

            root = Config().paths.projects_root
            path = os.path.abspath(os.path.join(*chain([root], sub_path)))
            prefix = sub_path[-1] if sub_path else ""
            if prefix == "":
                path += os.path.sep

            if root in path[:len(root)]:
                valid_parent_directory = os.path.isdir(os.path.dirname(path))
                if suffix and not path.endswith(suffix):
                    suffixed_path = path + suffix
                    exists = (os.path.isdir(suffixed_path) and is_directory or
                              os.path.isfile(suffixed_path) and not is_directory)

                else:
                    exists = os.path.isdir(path) and is_directory or os.path.isfile(path) and not is_directory

                if not valid_parent_directory:
                    reason = "Root directory does not exist"
                else:
                    reason = ""
            else:
                valid_parent_directory = False
                exists = False
                reason = "Path not allowed"

            if valid_parent_directory:
                suggestions = tuple("/".join(chain([command], os.path.relpath(p, root).split(os.sep)))
                                    for p in glob.glob(path + "*" + (suffix if is_directory else ""))
                                    if os.path.isdir(p) and safe_directory_name(os.path.basename(p)))

                suggestion_is_directories = tuple(1 for _ in suggestions)

                if not is_directory:

                    candidates = glob.glob(os.path.join(os.path.dirname(path), prefix + "*" + suffix))

                    suggestion_files = tuple("/".join(chain([command], os.path.relpath(p, root).split(os.sep)))
                                        for p in candidates if os.path.isfile(p))

                    suggestions = suggestion_files + suggestions

                    suggestion_is_directories = tuple(0 for _ in suggestion_files) + suggestion_is_directories

            else:
                suggestions = tuple()
                suggestion_is_directories = tuple()

            if check_has_analysis:
                has_analysis = path_has_saved_project_state(path)
            else:
                has_analysis = None

            return jsonify(path="/".join(chain([command], sub_path)), valid_parent=valid_parent_directory,
                           reason=reason,
                           suggestions=suggestions,
                           suggestion_is_directories=suggestion_is_directories,
                           prefix=prefix,
                           exists=exists,
                           has_analysis=has_analysis)

        return jsonify(path='/', valid_parent=False, reason="Path not allowed")

    # End of adding routes
