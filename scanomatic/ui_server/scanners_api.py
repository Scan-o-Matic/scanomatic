from __future__ import absolute_import
from flask import request, jsonify, Blueprint, current_app
from httplib import NOT_FOUND
from .general import json_abort

blueprint = Blueprint("scanners_api", __name__)


@blueprint.route("", methods=['GET'])
def scanners_get():
    get_free = request.args.get('free', False)
    scanners = current_app.config['scanners']
    return jsonify(
        scanners.get_free() if get_free else scanners.get_all()
    )


@blueprint.route("/<scanner>", methods=['GET'])
def scanner_get(scanner):
    scanners = current_app.config['scanners']
    if scanners.has_scanner(scanner):
        return jsonify(**scanners.get(scanner))
    return json_abort(
        NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
    )
