from __future__ import absolute_import
from datetime import datetime
from httplib import NOT_FOUND, OK

from flask import request, jsonify, Blueprint, current_app
from flask_restful import Api, Resource
from werkzeug.exceptions import NotFound

from .general import json_abort
from .serialization import job2json

blueprint = Blueprint("scanners_api", __name__)


@blueprint.route("", methods=['GET'])
def scanners_get():
    get_free = request.args.get('free', False)
    scanning_store = current_app.config['scanning_store']
    scanners = (
        scanning_store.get_free_scanners() if get_free else
        scanning_store.get_all_scanners()
    )
    return jsonify([scanner._asdict() for scanner in scanners])


@blueprint.route("/<scanner>", methods=['GET'])
def scanner_get(scanner):
    scanning_store = current_app.config['scanning_store']
    if scanning_store.has_scanner(scanner):
        return jsonify(**scanning_store.get_scanner(scanner)._asdict())
    return json_abort(
        NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
    )


@blueprint.route("/<scanner>/status", methods=['POST'])
def scanner_status_update(scanner):
    scanning_store = current_app.config['scanning_store']
    if not scanning_store.has_scanner(scanner):
        # TODO: should create non-existent
        return json_abort(
            NOT_FOUND, reason="Scanner '{}' unknown".format(
                scanner)
        )

    status = request.get_json(silent=True, force=True)
    scanning_store.scanner_status_update(scanner, status["message"])
    return "", OK



@blueprint.route("/<scanner>/status", methods=['GET'])
def scanner_status_get(scanner):
    scanning_store = current_app.config['scanning_store']
    if scanning_store.has_scanner(scanner):
        return jsonify(scanning_store.get_latest_scanner_status(scanner))
    return json_abort(
        NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
    )


class ScannerJob(Resource):
    def get(self, scannerid):
        db = current_app.config['scanning_store']
        if not db.has_scanner(scannerid):
            raise NotFound
        job = db.get_current_scanjob(scannerid, datetime.now())
        if job:
            return job2json(job)


api = Api(blueprint)
api.add_resource(ScannerJob, '/<scannerid>/job', endpoint='scanner-job')
