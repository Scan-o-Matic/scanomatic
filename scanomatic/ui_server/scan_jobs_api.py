from __future__ import absolute_import

from datetime import datetime, timedelta
from httplib import BAD_REQUEST, CONFLICT, CREATED, OK
from uuid import uuid1

from flask import Blueprint, jsonify, request
from flask_restful import Api, Resource
import pytz
from werkzeug.exceptions import BadRequest, NotFound

from scanomatic.models.scanjob import ScanJob
from scanomatic.scanning.delete_scanjob import (
    DeleteScanjobError, delete_scanjob
)
from scanomatic.scanning.exceptions import UnknownScanjobError
from scanomatic.scanning.terminate_scanjob import (
    TerminateScanJobError, terminate_scanjob
)
from . import database
from .general import json_abort
from .serialization import job2json

blueprint = Blueprint('scan_jobs_api', __name__)

# There must be a minimum interval for scanning to avoid start
# of scan while still scanning.
MINIMUM_INTERVAL = timedelta(minutes=5)


@blueprint.route("", methods=['POST'])
def scan_jobs_add():
    data_object = request.get_json()
    scanjobstore = database.getscanjobstore()
    scannerstore = database.getscannerstore()
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
    scanjobstore = database.getscanjobstore()
    return jsonify([
        job2json(job) for job in scanjobstore.get_all_scanjobs()
    ])


class ScanJobDocument(Resource):

    def get(self, scanjobid):
        scanjobstore = database.getscanjobstore()
        try:
            job = scanjobstore.get_scanjob_by_id(scanjobid)
        except KeyError:
            raise NotFound
        return job2json(job)

    def delete(self, scanjobid):
        scanjobstore = database.getscanjobstore()
        try:
            delete_scanjob(scanjobstore, scanjobid)
        except UnknownScanjobError:
            raise NotFound
        except DeleteScanjobError as e:
            raise BadRequest(str(e))


class ScanJobStartController(Resource):
    def post(self, scanjobid):
        scanjobstore = database.getscanjobstore()
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


class ScanJobTerminateController(Resource):

    def post(self, scanjobid):
        scanjobstore = database.getscanjobstore()
        data_obj = request.get_json(silent=True)
        if data_obj is not None:
            message = data_obj.get('message')
        else:
            message = None
        try:
            terminate_scanjob(scanjobstore, scanjobid, message)
        except UnknownScanjobError:
            raise NotFound('No scan job with id {}'.format(scanjobid))
        except TerminateScanJobError as e:
            raise BadRequest(str(e))
        return '', OK


api = Api(blueprint)
api.add_resource(ScanJobStartController, '/<scanjobid>/start')
api.add_resource(ScanJobTerminateController, '/<scanjobid>/terminate')
api.add_resource(ScanJobDocument, '/<scanjobid>')
