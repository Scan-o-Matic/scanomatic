from flask import request, jsonify, redirect
import time
import os
import sys
import signal
from subprocess import Popen, PIPE
from threading import Thread
from scanomatic import get_version
from scanomatic.io.mail import can_get_server_with_current_settings
from scanomatic.io.source import parse_version, upgrade, git_version, highest_version, get_source_information
from .general import decorate_api_access_restriction
_GIT_INFO = None
_GIT_INFO_RECHECK = 3600 * 24


def time_to_cache_git_info():

    return _GIT_INFO["check_time"] + _GIT_INFO_RECHECK < time.time()


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
    @decorate_api_access_restriction
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
    @decorate_api_access_restriction
    def _app_actions(action=None):

        if action == 'reboot':
            app.log_recycler.cancel()
            shutdown_server()
            Thread(target=relaunch).start()
            return jsonify(success=True)

        elif action == 'shutdown':
            app.log_recycler.cancel()
            shutdown_server()
            return jsonify(success=True)

        elif action == 'version':

            return jsonify(success=True, version=get_version(), version_ints=parse_version(get_version()),
                           source_information=get_source_information(test_info=True))

        elif action == 'upgradable':

            global _GIT_INFO

            if not _GIT_INFO or request.values.get('force_check', False, type=bool) or time_to_cache_git_info():
                git_ver_as_text = git_version()
                _GIT_INFO = {
                    "check_time": time.time(),
                    "version_text": git_ver_as_text,
                    "version_tuple": parse_version(git_ver_as_text),
                    "cached": False
                }
            else:
                _GIT_INFO["cached"] = True

            local_version = {
                    "check_time": time.time(),
                    "version_text": get_version(),
                    "version_tuple": parse_version(get_version()),
                    "cached": False
            }

            return jsonify(success=True, remote_version=_GIT_INFO, local_version=local_version,
                           upgradable=highest_version(local_version["version_tuple"],
                                                      _GIT_INFO["version_tuple"]) != local_version["version_tuple"])

        elif action == 'upgrade':

            branch = request.values.get('branch', None)
            success = upgrade(branch=branch)
            if success:
                return jsonify(success=True)
            else:
                return jsonify(success=False, reason="Could be no update is available or installation failed.")

        else:
            return jsonify(success=False, reason="Unknown action '{0}'".format(action))

    @app.route("/api/job/<job_id>/<job_command>")
    @decorate_api_access_restriction
    def _communicate_with_job(job_id="", job_command=""):

        if rpc_client.online:
            val = rpc_client.communicate(job_id, job_command)
            return jsonify(success=val, reason=None if val else "Refused by server")

        return jsonify(success=False, is_endpoint=True, reason="Server offline")

    @app.route("/api/settings/mail/possible")
    @decorate_api_access_restriction
    def can_possibly_mail():

        return jsonify(success=True, is_endpoint=True, can_possibly_mail=can_get_server_with_current_settings())

    @app.route("/api/power_manager/status")
    @decorate_api_access_restriction
    def get_pm_status():

        if rpc_client.online:
            val = rpc_client.get_power_manager_info()
            return jsonify(success=True, is_endpoint=True, **val)

        else:
            return jsonify(success=False, is_endpoint=True, reason="Server offline")

    @app.route("/api/power_manager/test")
    @decorate_api_access_restriction
    def redirect_to_pm():

        if rpc_client.online:
            val = rpc_client.get_power_manager_info()
            if val['host']:
                uri = val['host']
                if not uri.startswith("http"):
                    uri = "http://" + uri
                return redirect(uri)
            else:
                return jsonify(success=False, is_endpoint=True, reason="Power Manager not know/found by Scan-o-Matic. Check your settings.")

        else:
            return jsonify(success=False, is_endpoint=True, reason="Server offline")


            # END OF ADDING ROUTES
