from flask import request, jsonify
from .general import json_abort


def add_routes(app):

    @app.route("/api/scanners", methods=['GET'])
    def scanners_get():
        get_free = request.args.get('free', False)
        scanners = app.config['scanners']
        return jsonify(
            scanners=scanners.get_free() if get_free else scanners.get_all()
        )

    @app.route("/api/scanners/<scanner>", methods=['GET'])
    def scanner_get(scanner):
        scanners = app.config['scanners']
        if scanners.has_scanner(scanner):
            return jsonify(scanner=scanners.get(scanner))
        return json_abort(400, reason="Scanner '{}' unknown".format(scanner))
