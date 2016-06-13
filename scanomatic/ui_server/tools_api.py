from flask import request, Flask, jsonify
from itertools import product, chain
import os
import glob

from scanomatic.ui_server.general import safe_directory_name
from scanomatic.io.app_config import Config
from scanomatic.io.logger import Logger

_logger = Logger("Tools API")


def valid_range(settings):
    return 'min' in settings and 'max' in settings


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/tools/selection", methods=['POST'])
    @app.route("/api/tools/selection/", methods=['POST'])
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
        if request.json is None:
            return jsonify()

        if operation == 'separate':
            response = {}
            for key in request.json:

                settings = request.json.get(key)

                if valid_range(settings):
                    response[key] = range(settings['min'], settings['max'])

            return jsonify(**response)

        elif operation == 'rect':
            return jsonify(
                **{k: v for k, v in
                   zip(request.json,
                       zip(*product(*(range(v['min'], v['max'])
                                      for k, v in request.json.iteritems()
                                      if valid_range(v)))))})

        else:
            return jsonify()

    @app.route("/api/tools/coordinates", methods=['POST'])
    @app.route("/api/tools/coordinates/", methods=['POST'])
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
        if operation == 'create':
            keys = request.json.get('keys', request.json.keys())
            return jsonify(coordinates=zip(*(request.json[k] for k in keys)))

        elif operation == 'parse':
            return jsonify(selection=zip(*request.json['coordinates']))

    @app.route("/api/tools/path/")
    @app.route("/api/tools/path/<command>", methods=['get', 'post'])
    @app.route("/api/tools/path/<command>/", methods=['get', 'post'])
    @app.route("/api/tools/path/<command>/<path:sub_path>", methods=['get', 'post'])
    def _experiment_commands(command=None, sub_path=""):

        if command is None:
            command = 'root'

        sub_path = sub_path.split("/")
        try:
            is_directory = bool(request.values.get('isDirectory', type=int, default=True))
        except ValueError:
            is_directory = True

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
                if not is_directory:
                    suggestions = tuple("/".join(chain([command], os.path.relpath(p, root).split(os.sep)))
                                        for p in glob.glob(os.path.join(os.path.dirname(path), prefix + "*" + suffix))
                                        if os.path.isfile(p)) + suggestions

            else:
                suggestions = tuple()

            _logger.info("{0}: {1}".format(path, glob.glob(path + "*")))

            return jsonify(path="/".join(chain([command], sub_path)), valid_parent=valid_parent_directory,
                           reason=reason, suggestions=suggestions, prefix=prefix, exists=exists)

        return jsonify(path='/', valid_parent=False, reason="Path not allowed")

    # End of adding routes
