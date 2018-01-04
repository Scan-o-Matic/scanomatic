from flask import jsonify, redirect
from scanomatic import get_version
from scanomatic.io.mail import can_get_server_with_current_settings
from scanomatic.io.source import parse_version, get_source_information
from .general import json_abort
from xmlrpclib import Fault


def add_routes(app, rpc_client):

    @app.route("/api/app/version", methods=['get'])
    def _app_version():
        return jsonify(
            version=get_version(),
            version_ints=parse_version(get_version()),
            source_information=get_source_information(test_info=True)
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

    @app.route("/api/settings/mail/possible")
    def can_possibly_mail():

        return jsonify(
            can_possibly_mail=can_get_server_with_current_settings()
        )

    @app.route("/api/power_manager/status")
    def get_pm_status():

        if rpc_client.online:
            try:
                val = rpc_client.get_power_manager_info()
            except Fault:
                return json_abort(
                    400,
                    reason="Unexpected error, try again in a bit.")

            return jsonify(**val)

        else:
            return json_abort(400, reason="Server offline")

    @app.route("/api/power_manager/test")
    def redirect_to_pm():

        if rpc_client.online:
            val = rpc_client.get_power_manager_info()
            if 'host' in val and val['host']:
                uri = val['host']
                if not uri.startswith("http"):
                    uri = "http://" + uri
                return redirect(uri)
            else:
                return json_abort(
                    400,
                    reason="Power Manager not know/found by Scan-o-Matic. Check your settings."
                )

        else:
            return json_abort(400, reason="Server offline")

            # END OF ADDING ROUTES
