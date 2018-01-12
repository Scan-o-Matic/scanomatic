from __future__ import absolute_import
from flask import request, jsonify, Blueprint, current_app
from httplib import NOT_FOUND
from .general import json_abort

blueprint = Blueprint("scanners_api", __name__)


@blueprint.route("", methods=['GET'])
def scanners_get():
    get_free = request.args.get('free', False)
    scanning_store = current_app.config['scanning_store']
    return jsonify(
        scanning_store.get_free_scanners() if get_free else
        scanning_store.get_all_scanners()
    )


@blueprint.route("/<scanner>", methods=['GET'])
def scanner_get(scanner):
    scanning_store = current_app.config['scanning_store']
    if scanning_store.has_scanner(scanner):
        return jsonify(**scanning_store.get_scanner(scanner))
    return json_abort(
        NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
    )
