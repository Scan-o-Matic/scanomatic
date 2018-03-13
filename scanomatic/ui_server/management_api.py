from __future__ import absolute_import

from flask import jsonify
from scanomatic import get_version
from scanomatic.io.mail import can_get_server_with_current_settings
from scanomatic.io.source import parse_version
from .general import json_abort


def add_routes(app, rpc_client):

    @app.route("/api/app/version", methods=['get'])
    def _app_version():
        return jsonify(
            version=get_version(),
            version_ints=parse_version(get_version()),
        )

    @app.route("/api/job/<job_id>/<job_command>")
    def _communicate_with_job(job_id="", job_command=""):

        if rpc_client.online:
            val = rpc_client.communicate(job_id, job_command)
            if val:
                return jsonify()
            else:
                return json_abort(
                    400,
                    reason=None if val else "Refused by server"
                )

        return json_abort(400, reason="Server offline")

        # END OF ADDING ROUTES
