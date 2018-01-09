from flask import send_from_directory, jsonify

from scanomatic.io.paths import Paths
from .general import convert_path_to_url, json_abort


def add_routes(app, rpc_client):

    @app.route("/api/status/<status_type>")
    @app.route("/api/status/<status_type>/<status_query>")
    def _status_api(status_type="", status_query=None):

        if status_type != "" and not rpc_client.online:
            return jsonify(success=False, reason="Server offline")

        if status_type == 'queue':
            return jsonify(queue=rpc_client.get_queue_status())
        elif 'scanners' == status_type:
            if status_query is None or status_query.lower() == 'all':
                return jsonify(scanners=[{'scanner_name': 'Test'}])
            elif status_query.lower() == 'free':
                return jsonify(scanners={1: 'Test'})
            else:
                return jsonify(
                    scanner={
                        'scanner_name': 'Test',
                        'socket': 1,
                        'power': True,
                        'owner': None,
                    }
                )
        elif 'jobs' == status_type:
            data = rpc_client.get_job_status()
            for item in data:
                if item['type'] == "Feature Extraction Job":
                    item['label'] = convert_path_to_url("", item['label'])
                if 'log_file' in item and item['log_file']:
                    item['log_file'] = convert_path_to_url(
                        "/logs/project", item['log_file'])
            return jsonify(jobs=data)
        elif status_type == 'server':
            return jsonify(**rpc_client.get_status())
        else:
            return json_abort(reason='Unknown status request')
