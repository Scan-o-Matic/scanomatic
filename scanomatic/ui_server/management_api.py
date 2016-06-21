from flask import request, jsonify
import time
import os
import sys
import signal
from subprocess import Popen, PIPE
from threading import Thread
from scanomatic import get_version
from scanomatic.io.source import parse_version


def relaunch():

    time.sleep(1)
    os.execv(sys.argv[0], sys.argv)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


def add_routes(app, rpc_client):

    @app.route("/api/server/<action>", methods=['post', 'get'])
    def _server_actions(action=None):

        if action == 'reboot':

            if rpc_client.local and (request.args.get('force') == '1' or not rpc_client.working_on_job_or_has_queue):

                rpc_client.shutdown()
                time.sleep(5)
                t = time.time()
                while rpc_client.online and time.time() - t < 370:
                    time.sleep(0.5)
                if rpc_client.online:
                    return jsonify(success=False, reason="Failed to close the server")
                rpc_client.launch_local()
                t = time.time()
                while not rpc_client.online and time.time() - t < 60:
                    time.sleep(0.5)
                if rpc_client.online:
                    return jsonify(success=True)
                else:
                    return jsonify(success=False, reason="Failed to restart the server")

        elif action == 'shutdown':

            rpc_client.shutdown()
            t = time.time()
            while rpc_client.online and time.time() - t < 370:
                time.sleep(0.5)
            if rpc_client.online:
                return jsonify(success=False, reason="Failed to shut down server")
            return jsonify(success=True)

        elif action == 'launch':

            if rpc_client.online:
                return jsonify(success=False, reason="Server is already running")

            rpc_client.launch_local()
            t = time.time()
            while not rpc_client.online and time.time() - t < 60:
                time.sleep(0.5)
            if rpc_client.online:
                return jsonify(success=True)
            else:
                return jsonify(success=False, reason="Could not launch the server")

        elif action == 'kill':

            p = Popen(["ps", "-A"], stdout=PIPE)
            stdout, _ = p.communicate()
            server_ids = set()

            for server_proc in (proc for proc in stdout if "SoM Server" in proc):
                try:
                    proc_id = int(server_proc.strip().split(" ")[0])
                except ValueError:
                    return jsonify(success=False, reason="Could not parse process id of '{0}'".format(server_proc))

                os.kill(proc_id, signal.SIGKILL)
                server_ids.add(proc_id)

            if rpc_client.online:
                return jsonify(success=False,
                               reason="Tried to kill processes {0}, but somehow server is online".format(server_ids))

            return jsonify(success=True)

    @app.route("/api/app/<action>", methods=['post', 'get'])
    def _app_actions(action=None):

        if action == 'reboot':
            shutdown_server()
            Thread(target=relaunch).start()
            return jsonify(success=True)

        elif action == 'shutdown':
            shutdown_server()
            return jsonify(success=True)

        elif action == 'version':

            return jsonify(success=True, version=get_version(), version_ints=parse_version(get_version()))

        elif action == 'upgradable':

            # TODO: Only check if exists
            # TODO: Disallow frequent checks
            return jsonify(success=False, reason="Not implemented")

        elif action == 'upgrade':
            # TODO: Add upgrade from API
            return jsonify(success=False, reason="Not implemented")

    @app.route("/api/job/<job_id>/<job_command>")
    def _communicate_with_job(job_id="", job_command=""):

        if rpc_client.online:
            val = rpc_client.communicate(job_id, job_command)
            return jsonify(success=val, reason=None if val else "Refused by server")

        return jsonify(success=False, reason="Server offline")

    # END OF ADDING ROUTES
