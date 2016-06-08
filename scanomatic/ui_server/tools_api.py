from flask import request, Flask, jsonify
from itertools import product


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
