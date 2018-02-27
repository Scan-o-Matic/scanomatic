# coding: utf8
from __future__ import absolute_import
import httplib as HTTPStatus
from io import BytesIO

from flask import Blueprint, current_app, send_file
from flask_restful import Api, Resource, reqparse, inputs
from werkzeug.datastructures import FileStorage

from . import database
from .general import json_abort
from .serialization import scan2json
from scanomatic.models.scan import Scan
from scanomatic.util.scanid import generate_scan_id


class ScanCollection(Resource):
    def get(self):
        scanstore = database.getscanstore()
        return [scan2json(md) for md in scanstore.get_all_scans()]

    def post(self):
        scanjobstore = database.getscanjobstore()
        scanstore = database.getscanstore()
        imagestore = current_app.config['imagestore']
        parser = reqparse.RequestParser()
        parser.add_argument(
            'startTime',
            dest='start_time',
            type=inputs.datetime_from_iso8601,
            required=True,
        )
        parser.add_argument(
            'endTime',
            dest='end_time',
            type=inputs.datetime_from_iso8601,
            required=True,
        )
        parser.add_argument(
            'scanJobId',
            dest='scanjob_id',
            type=inputs.regex(r'\w+'),
            required=True,
        )
        parser.add_argument(
            'digest',
            type=inputs.regex(r'sha256:\w+'),
            required=True,
        )
        parser.add_argument(
            'image', type=FileStorage, location='files', required=True,
        )
        args = parser.parse_args(strict=True)
        try:
            scanjob = scanjobstore.get_scanjob_by_id(args.scanjob_id)
        except KeyError:
            return json_abort(
                HTTPStatus.BAD_REQUEST,
                message='Unknown scan job',
            )
        scan = Scan(
            id=generate_scan_id(scanjob, args.start_time),
            start_time=args.start_time,
            end_time=args.end_time,
            digest=args.digest,
            scanjob_id=args.scanjob_id,
        )
        # ⚠ Without support for transactions, the order of the following calls
        # matter: we add the metadata to the database only if the image has
        # been stored on the disk. That way, the worst that can happen is
        # an orphan image file. Otherwise, the database may thing it has
        # the scan whereas the image is not there. (I.e. the source of truth
        # is the database).
        imagestore.put(args.image.read(), scan)
        scanstore.add_scan(scan)
        return {'identifier': scan.id}, HTTPStatus.CREATED


class ScanDocument(Resource):
    def get(self, scanid):
        scanstore = database.getscanstore()
        try:
            return scan2json(scanstore.get_scan_by_id(scanid))
        except KeyError:
            return json_abort(
                HTTPStatus.NOT_FOUND, message='No scan with this id',
            )


class ScanImage(Resource):
    def get(self, scanid):
        scanstore = database.getscanstore()
        imagestore = current_app.config['imagestore']
        try:
            scan = scanstore.get_scan_by_id(scanid)
        except KeyError:
            return json_abort(
                HTTPStatus.NOT_FOUND, message='No scan with this id',
            )
        image = imagestore.get(scan)
        return send_file(BytesIO(image), mimetype='image/tiff')


blueprint = Blueprint('scans-api', __name__)
api = Api(blueprint)
api.add_resource(ScanCollection, '')
api.add_resource(ScanDocument, '/<scanid>')
api.add_resource(ScanImage, '/<scanid>/image')
