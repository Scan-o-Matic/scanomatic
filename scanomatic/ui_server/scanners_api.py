from __future__ import absolute_import
from datetime import datetime, timedelta
from httplib import NOT_FOUND, OK, BAD_REQUEST, CREATED, INTERNAL_SERVER_ERROR

from flask import request, jsonify, Blueprint, current_app
from flask_restful import Api, Resource, reqparse, inputs
import pytz
from werkzeug.exceptions import NotFound

from .general import json_abort
from .serialization import job2json, scanner_status2json, scanner2json
from scanomatic.io.scanning_store import (
    Scanner, DuplicateNameError)
from scanomatic.models.scannerstatus import ScannerStatus
from scanomatic.util.generic_name import get_generic_name

blueprint = Blueprint("scanners_api", __name__)
SCANNER_TIMEOUT = timedelta(minutes=5)


def _scanner_is_online(scanner_id, scanning_store):
    try:
        return (
            datetime.now(pytz.utc)
            - scanning_store.get_latest_scanner_status(
                scanner_id).server_time
            < SCANNER_TIMEOUT
        )
    except AttributeError:
        return False


@blueprint.route("", methods=['GET'])
def scanners_get():
    get_free = request.args.get('free', False)
    scanning_store = current_app.config['scanning_store']
    scanners = (
        scanning_store.get_free_scanners() if get_free else
        scanning_store.get_all_scanners()
    )
    return jsonify([
        scanner2json(
            scanner,
            _scanner_is_online(scanner.identifier, scanning_store)
        ) for scanner in scanners
    ])


@blueprint.route("/<scanner>", methods=['GET'])
def scanner_get(scanner):
    scanning_store = current_app.config['scanning_store']

    if scanning_store.has_scanner(scanner):
        return jsonify(scanner2json(
            scanning_store.get_scanner(scanner),
            _scanner_is_online(scanner, scanning_store)
        ))

    return json_abort(
        NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
    )


@blueprint.route("/<scanner>/status", methods=['PUT'])
def scanner_status_update(scanner):
    scanning_store = current_app.config['scanning_store']

    def _add_scanner(scanner_id):
        try:
            name = get_generic_name()
            scanning_store.add_scanner(Scanner(name, scanner_id))
        except DuplicateNameError:
            return json_abort(
                INTERNAL_SERVER_ERROR,
                reason="Failed to create scanner, please try again"
            )

    parser = reqparse.RequestParser()
    parser.add_argument('job', required=True)
    parser.add_argument(
        'startTime',
        dest='start_time',
        type=inputs.datetime_from_iso8601,
        required=True,
    )
    parser.add_argument(
        'imagesToSend',
        dest='images_to_send',
        type=inputs.natural,
        required=True,
    )
    args = parser.parse_args(strict=True)
    print(args.start_time)

    if not scanning_store.has_scanner(scanner):
        _add_scanner(scanner)
        status_code = CREATED
    else:
        status_code = OK

    status = request.get_json()
    try:
        scanning_store.add_scanner_status(
            scanner, ScannerStatus(server_time=datetime.now(pytz.utc), **args)
        )
    except KeyError:
        return json_abort(
            BAD_REQUEST,
            reason="Got malformed status '{}'".format(status)
        )

    return "", status_code


@blueprint.route("/<scanner>/status", methods=['GET'])
def scanner_status_get(scanner):
    scanning_store = current_app.config['scanning_store']

    if scanning_store.has_scanner(scanner):
        status = scanning_store.get_latest_scanner_status(scanner)

        if status is None:
            return jsonify({})

        return jsonify(scanner_status2json(status))

    return json_abort(
        NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
    )


class ScannerJob(Resource):
    def get(self, scannerid):
        db = current_app.config['scanning_store']
        if not db.has_scanner(scannerid):
            raise NotFound
        job = db.get_current_scanjob(scannerid, datetime.now(pytz.utc))
        if job:
            return job2json(job)


api = Api(blueprint)
api.add_resource(ScannerJob, '/<scannerid>/job', endpoint='scanner-job')
