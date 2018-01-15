from __future__ import absolute_import
from flask import request, jsonify, Blueprint, current_app
from uuid import uuid1
from httplib import BAD_REQUEST, FORBIDDEN, INTERNAL_SERVER_ERROR, CREATED

from scanomatic.io.scanning_store import ScanJobCollisionError, ScanJob
from .general import json_abort

blueprint = Blueprint('scan_jobs_api', __name__)

# There must be a minimum interval for scanning to avoid start
# of scan while still scanning.
MINIMUM_INTERVAL = 5


@blueprint.route("", methods=['POST'])
def scan_jobs_add():
    data_object = request.get_json()
    scanning_store = current_app.config['scanning_store']
    name = data_object.get('name', None)
    if not name:
        return json_abort(BAD_REQUEST, reason="No name supplied")
    if scanning_store.exists_scanjob_with('name', name):
        return json_abort(
            FORBIDDEN,
            reason="Name '{}' duplicated".format(name)
        )

    duration = data_object.get('duration', None)
    if not duration:
        return json_abort(BAD_REQUEST, reason="Duration not supplied")

    interval = data_object.get('interval', None)
    if not interval:
        return json_abort(BAD_REQUEST, reason="Interval not supplied")

    if interval < MINIMUM_INTERVAL:
        return json_abort(BAD_REQUEST, reason="Interval too short")

    scanner = data_object.get('scannerId', None)
    if scanner is None:
        return json_abort(BAD_REQUEST, reason="Scanner not supplied")
    if not scanning_store.has_scanner(scanner):
        return json_abort(
            BAD_REQUEST,
            reason="Scanner '{}' unknown".format(scanner)
        )

    identifier = uuid1().hex
    try:
        scanning_store.add_scanjob(ScanJob(
            identifier=identifier,
            name=name,
            duration=duration,
            interval=interval,
            scanner_id=scanner
        ))
    except ScanJobCollisionError:
        return json_abort(INTERNAL_SERVER_ERROR, reason="Identifier collision")

    return jsonify(jobId=identifier), CREATED


@blueprint.route("", methods=['GET'])
def scan_jobs_list():
    def py2js(data):
        return {
            'identifier': data.identifier,
            'name': data.name,
            'duration': data.duration,
            'interval': data.interval,
            'scannerId': data.scanner_id,
        }

    scanning_store = current_app.config['scanning_store']
    return jsonify([
        py2js(job) for job in scanning_store.get_all_scanjobs()
    ])
