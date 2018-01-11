from __future__ import absolute_import
from flask import request, jsonify
from uuid import uuid1

from scanomatic.io.scan_jobs import ScanNameCollision
from scanomatic.io.logger import Logger
from .general import json_abort

_LOGGER = Logger("Experiment/Project API")


def add_routes(app):

    @app.route("/api/scan-jobs", methods=['POST'])
    def _experiment_api_add():
        data_object = request.get_json(silent=True, force=True)
        _LOGGER.info("Experiment json {}".format(data_object))
        scan_jobs = app.config['scan_jobs']
        name = data_object.get('name', None)
        if not name:
            return json_abort(400, reason="No name supplied")
        if scan_jobs.exists_job_with('name', name):
            return json_abort(400, reason="Name '{}' duplicated".format(name))

        duration = data_object.get('duration', None)
        if not duration:
            return json_abort(400, reason="Duration not supplied")

        interval = data_object.get('interval', None)
        if not interval:
            return json_abort(400, reason="Interval not supplied")

        if interval < 5:
            return json_abort(400, reason="Interval too short")

        scanners = app.config['scanners']
        scanner = data_object.get('scannerName', None)
        if scanner is None:
            return json_abort(400, reason="Scanner not supplied")
        if not scanners.has_scanner(scanner):
            return json_abort(
                400,
                reason="Scanner '{}' unknown".format(scanner)
            )

        identifier = uuid1().hex
        try:
            scan_jobs.add_job(identifier, {
                'id': identifier,
                'name': name,
                'duration': duration,
                'interval': interval,
                'scanner': scanner
            })
        except ScanNameCollision:
            return json_abort(400, reason="Identifier collision")

        return jsonify(jobId=identifier)

    @app.route("/api/scan-jobs", methods=['GET'])
    def _experiment_api_list():

        scanners = app.config['scanners']
        scan_jobs = app.config['scan_jobs']
        return jsonify(jobs=[
            dict(job, scanner=scanners.get(job['scanner']))
            for job in scan_jobs.get_jobs()
        ])
