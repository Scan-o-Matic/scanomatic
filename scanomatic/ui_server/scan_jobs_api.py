from __future__ import absolute_import
from datetime import datetime, timedelta
from httplib import BAD_REQUEST, CONFLICT, INTERNAL_SERVER_ERROR, CREATED, OK
from uuid import uuid1

from flask import request, jsonify, Blueprint, current_app
from flask_restful import Api, Resource
import pytz
from werkzeug.exceptions import NotFound

from scanomatic.io.scanning_store import (
    ScanJobCollisionError, ScanJobUnknownError,
)
from scanomatic.models.scanjob import ScanJob
from .general import json_abort
from .serialization import job2json


blueprint = Blueprint('scan_jobs_api', __name__)

# There must be a minimum interval for scanning to avoid start
# of scan while still scanning.
MINIMUM_INTERVAL = timedelta(minutes=5)


@blueprint.route("", methods=['POST'])
def scan_jobs_add():
    data_object = request.get_json()
    scanning_store = current_app.config['scanning_store']
    name = data_object.get('name', None)
    if not name:
        return json_abort(BAD_REQUEST, reason="No name supplied")
    if scanning_store.exists_scanjob_with('name', name):
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
    scanning_store = current_app.config['scanning_store']
    return jsonify([
        job2json(job) for job in scanning_store.get_all_scanjobs()
    ])


class ScanJobDocument(Resource):
    def get(self, scanjobid):
        db = current_app.config['scanning_store']
        try:
            job = db.get_scanjob(scanjobid)
        except ScanJobUnknownError:
            raise NotFound
        return job2json(job)


class ScanJobStartController(Resource):
    def post(self, scanjobid):
        db = current_app.config['scanning_store']
        try:
            job = db.get_scanjob(scanjobid)
        except ScanJobUnknownError:
            raise NotFound
        now = datetime.now(pytz.utc)
        if job.start_time is not None:
            return json_abort(CONFLICT, reason='Scanning Job already started')
        if db.has_current_scanjob(job.scanner_id, now):
            return json_abort(CONFLICT, reason='Scanner busy')
        db.update_scanjob(job._replace(start_time=now))
        return '', OK


api = Api(blueprint)
api.add_resource(ScanJobStartController, '/<scanjobid>/start')
api.add_resource(ScanJobDocument, '/<scanjobid>')
