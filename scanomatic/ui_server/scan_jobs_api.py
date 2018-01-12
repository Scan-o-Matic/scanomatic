from __future__ import absolute_import
from flask import request, jsonify, Blueprint, current_app
from uuid import uuid1
from httplib import BAD_REQUEST, FORBIDDEN, INTERNAL_SERVER_ERROR, CREATED

from scanomatic.io.scan_jobs import ScanJobCollisionError, ScanJob
from scanomatic.io.logger import Logger
from .general import json_abort

blueprint = Blueprint('scan_jobs_api', __name__)

# There must be a minimum interval for scanning to avoid start
# of scan while still scanning.
MINIMUM_INTERVAL = 5


@blueprint.route("", methods=['POST'])
def scan_jobs_add():
    data_object = request.get_json()
    scan_jobs = current_app.config['scan_jobs']
    name = data_object.get('name', None)
    if not name:
        return json_abort(BAD_REQUEST, reason="No name supplied")
    if scan_jobs.exists_job_with('name', name):
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

    scanners = current_app.config['scanners']
    scanner = data_object.get('scannerName', None)
    if scanner is None:
        return json_abort(BAD_REQUEST, reason="Scanner not supplied")
    if not scanners.has_scanner(scanner):
        return json_abort(
            BAD_REQUEST,
            reason="Scanner '{}' unknown".format(scanner)
        )

    identifier = uuid1().hex
    try:
        scan_jobs.add_job(ScanJob(
            identifier=identifier,
            name=name,
            duration=duration,
            interval=interval,
            scanner=scanner
        ))
    except ScanJobCollisionError:
        return json_abort(INTERNAL_SERVER_ERROR, reason="Identifier collision")

    return jsonify(jobId=identifier), CREATED


@blueprint.route("", methods=['GET'])
def scan_jobs_list():

    scanners = current_app.config['scanners']
    scan_jobs = current_app.config['scan_jobs']
    return jsonify([
        dict(job._asdict(), scanner=scanners.get(job.scanner))
        for job in scan_jobs.get_jobs()
    ])
