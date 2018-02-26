from __future__ import absolute_import
from datetime import datetime, timedelta
from httplib import NOT_FOUND, OK, CREATED, INTERNAL_SERVER_ERROR

from flask import jsonify, Blueprint, current_app
from flask_restful import Api, Resource, reqparse, inputs
import pytz
from werkzeug.exceptions import NotFound

from . import database
from .general import json_abort
from .serialization import job2json, scanner_status2json, scanner2json
from scanomatic.scanning.update_scanner_status import (
    update_scanner_status, UpdateScannerStatusError,
)

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
    scannerstore = database.getscannerstore()
    scanning_store = current_app.config['scanning_store']
    scanners = scannerstore.get_all()
    return jsonify([
        scanner2json(
            scanner,
            _scanner_is_online(scanner.identifier, scanning_store)
        ) for scanner in scanners
    ])


@blueprint.route("/<scanner>", methods=['GET'])
def scanner_get(scanner):
    scannerstore = database.getscannerstore()
    scanning_store = current_app.config['scanning_store']
    try:
        return jsonify(scanner2json(
            scannerstore.get_scanner_by_id(scanner),
            _scanner_is_online(scanner, scanning_store)
        ))
    except KeyError:
        return json_abort(
            NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
        )


@blueprint.route("/<scanner>/status", methods=['PUT'])
def scanner_status_update(scanner):
    scanning_store = current_app.config['scanning_store']
    scannerstore = database.getscannerstore()
    parser = reqparse.RequestParser()
    parser.add_argument('job')
    parser.add_argument(
        'startTime',
        dest='start_time',
        type=inputs.datetime_from_iso8601,
        required=True,
    )
    parser.add_argument(
        'nextScheduledScan',
        dest='next_scheduled_scan',
        type=inputs.datetime_from_iso8601,
    )
    parser.add_argument(
        'imagesToSend',
        dest='images_to_send',
        type=inputs.natural,
        required=True,
    )
    parser.add_argument(
        'devices',
        dest='devices',
        action='append',
    )
    args = parser.parse_args(strict=True)
    try:
        result = update_scanner_status(
            scannerstore, scanning_store, scanner, **args)
    except UpdateScannerStatusError as error:
        return json_abort(INTERNAL_SERVER_ERROR, reason=str(error))
    status_code = CREATED if result.new_scanner else OK
    return "", status_code


@blueprint.route("/<scanner>/status", methods=['GET'])
def scanner_status_get(scanner):
    scanning_store = current_app.config['scanning_store']
    scannerstore = database.getscannerstore()
    if not scannerstore.has_scanner_with_id(scanner):
        return json_abort(
            NOT_FOUND, reason="Scanner '{}' unknown".format(scanner)
        )
    status = scanning_store.get_latest_scanner_status(scanner)
    if status is None:
        return jsonify({})
    return jsonify(scanner_status2json(status))


class ScannerJob(Resource):
    def get(self, scannerid):
        scanjobstore = database.getscanjobstore()
        scannerstore = database.getscannerstore()
        if not scannerstore.has_scanner_with_id(scannerid):
            raise NotFound
        job = scanjobstore.get_current_scanjob_for_scanner(
            scannerid, datetime.now(pytz.utc))
        if job:
            return job2json(job)


api = Api(blueprint)
api.add_resource(ScannerJob, '/<scannerid>/job', endpoint='scanner-job')
