from __future__ import absolute_import
from datetime import datetime, timedelta
from httplib import BAD_REQUEST, CONFLICT, CREATED, OK
from uuid import uuid1

from flask import request, jsonify, Blueprint
from flask_restful import Api, Resource
import pytz
from werkzeug.exceptions import NotFound

from scanomatic.data.scannerstore import ScannerStore
from scanomatic.data.scanjobstore import ScanJobStore
from scanomatic.models.scanjob import ScanJob
from .general import json_abort
from .serialization import job2json
from . import database


blueprint = Blueprint('scan_jobs_api', __name__)

# There must be a minimum interval for scanning to avoid start
# of scan while still scanning.
MINIMUM_INTERVAL = timedelta(minutes=5)


@blueprint.route("", methods=['POST'])
def scan_jobs_add():
    data_object = request.get_json()
    scanjobstore = ScanJobStore(database.connect())
    scannerstore = ScannerStore(database.connect())
    name = data_object.get('name', None)
    if not name:
        return json_abort(BAD_REQUEST, reason="No name supplied")
    if scanjobstore.has_scanjob_with_name(name):
        return json_abort(
            CONFLICT,
            reason="Name '{}' duplicated".format(name)
        )

    duration = data_object.get('duration', None)
    if not duration:
        return json_abort(BAD_REQUEST, reason="Duration not supplied")
    try:
        duration = timedelta(seconds=duration)
    except TypeError:
        return json_abort(BAD_REQUEST, reason="Invalid duration")

    interval = data_object.get('interval', None)
    if not interval:
        return json_abort(BAD_REQUEST, reason="Interval not supplied")
    try:
        interval = timedelta(seconds=interval)
    except TypeError:
        return json_abort(BAD_REQUEST, reason="Invalid interval")

    if interval < MINIMUM_INTERVAL:
        return json_abort(BAD_REQUEST, reason="Interval too short")

    scanner = data_object.get('scannerId', None)
    if scanner is None:
        return json_abort(BAD_REQUEST, reason="Scanner not supplied")
    if not scannerstore.has_scanner_with_id(scanner):
        return json_abort(
            BAD_REQUEST,
            reason="Scanner '{}' unknown".format(scanner)
        )

    identifier = uuid1().hex
    scanjobstore.add_scanjob(ScanJob(
        identifier=identifier,
        name=name,
        duration=duration,
        interval=interval,
        scanner_id=scanner
    ))

    return jsonify(identifier=identifier), CREATED


@blueprint.route("", methods=['GET'])
def scan_jobs_list():
    scanjobstore = ScanJobStore(database.connect())
    return jsonify([
        job2json(job) for job in scanjobstore.get_all_scanjobs()
    ])


class ScanJobDocument(Resource):
    def get(self, scanjobid):
        scanjobstore = ScanJobStore(database.connect())
        try:
            job = scanjobstore.get_scanjob_by_id(scanjobid)
        except KeyError:
            raise NotFound
        return job2json(job)


class ScanJobStartController(Resource):
    def post(self, scanjobid):
        scanjobstore = ScanJobStore(database.connect())
        try:
            job = scanjobstore.get_scanjob_by_id(scanjobid)
        except KeyError:
            raise NotFound
        now = datetime.now(pytz.utc)
        if job.start_time is not None:
            return json_abort(CONFLICT, reason='Scanning Job already started')
        if scanjobstore.get_current_scanjob_for_scanner(job.scanner_id, now):
            return json_abort(CONFLICT, reason='Scanner busy')
        scanjobstore.set_scanjob_start_time(scanjobid, now)
        return '', OK


api = Api(blueprint)
api.add_resource(ScanJobStartController, '/<scanjobid>/start')
api.add_resource(ScanJobDocument, '/<scanjobid>')
