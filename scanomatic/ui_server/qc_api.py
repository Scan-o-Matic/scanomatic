from __future__ import absolute_import

from datetime import datetime
from enum import Enum
from glob import glob
from itertools import chain, product
import os
import re
from subprocess import call
import time
import uuid

from dateutil import tz
from flask import Flask, jsonify, request, send_from_directory, Blueprint
from werkzeug.datastructures import FileStorage

from scanomatic.data_processing import phenotyper
from scanomatic.data_processing import analysis_loader
from scanomatic.data_processing.norm import Offsets, infer_offset
from scanomatic.data_processing.phenotypes import (
    PhenotypeDataType, get_sort_order, infer_phenotype_from_name
)
from scanomatic.generics.phenotype_filter import Filter
from scanomatic.io.app_config import Config
from scanomatic.io.paths import Paths
from scanomatic.ui_server.general import (
    convert_path_to_url, convert_url_to_path, get_project_name,
    get_search_results, json_response, serve_zip_file, json_abort,
)

RESERVATION_TIME = 60 * 5
FILM_TYPES = {'colony': 'animate_colony_growth("{save_target}", {pos}, "{path}")',
              'detection': 'animate_blob_detection("{save_target}", {pos}, "{path}")',
              '3d': 'animate_3d_colony("{save_target}", {pos}, "{path}")'}

blueprint = Blueprint('qc_api', __name__)


@blueprint.route(
    "/growthcurves/<int:plate>/<path:project>",
    methods=['GET'],
)
def get_plate_curves_and_time(plate, project):
    loader = analysis_loader.AnalysisLoader(project)
    try:
        times_data = loader.times
    except analysis_loader.CorruptAnalysisError:
        return json_abort(
            400,
            reason="Could not locate scan times.",
        )

    try:
        plate_data = loader.get_plate_data(plate)
    except analysis_loader.CorruptAnalysisError:
        return json_abort(
            400,
            reason="Could not locate growth curves data files.",
        )
    except analysis_loader.PlateNotFoundError:
        return json_abort(
            400,
            reason='Plate not part of experiment.'
        )

    return jsonify(
        times_data=times_data.tolist(),
        raw_data=plate_data.raw.tolist(),
        smooth_data=plate_data.smooth.tolist(),
    )

class LockState(Enum):

    Unlocked = 0
    """:type: LockState"""
    LockedByOther = 1
    """:type: LockState"""
    LockedByMe = 2
    """:type: LockState"""
    LockedByMeTemporary = 3
    """:type: LockState"""


def owns_lock(lock_state):

    return lock_state is LockState.LockedByMe or lock_state is LockState.LockedByMeTemporary


def _get_state_update_response(path, response, success=None):

    try:
        state = phenotyper.Phenotyper.LoadFromState(path)
    except ImportError:
        name = None
        success = False
        state = None
        response['reason'] = "Feature extraction outdated, please re-run."
    else:
        name = get_project_name(path)
        print("Loaded state '{0}' from: {1}".format(name, path))

    response.update({'is_endpoint': True, 'is_project': True, 'project_name': name})

    if success is not None:
        response['success'] = success

    return state, name


def _make_film(film_type, save_target=None, pos=None, path=None):
    code = FILM_TYPES[film_type].format(save_target=save_target, pos=pos, path=path)
    return_code = call(['python', '-c', 'from scanomatic.qc import analysis_results;analysis_results.{0}'.format(code)])
    return return_code == 0


def _get_key():

    key = uuid.uuid4().hex
    return key


def _update_lock(lock_file_path, key, ip):

    with open(lock_file_path, 'w') as fh:
        fh.write("|".join((str(time.time()), str(key), str(ip))))
    return True


def _remove_lock(path):

    lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)
    os.remove(lock_file_path)

    return True


def _parse_lock_file(data):

    try:
        time_stamp, current_key, ip = data.split("|")
    except ValueError:
        try:
            time_stamp, current_key = data.split("|")
            ip = ""
        except ValueError:
            time_stamp = 0
            current_key = ""
            ip = ""

    try:
        time_stamp = float(time_stamp)
    except ValueError:
        time_stamp = 0

    return time_stamp, current_key, ip


def _read_lock_file(path):

    lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)
    try:
        with open(lock_file_path, 'r') as fh:
            time_stamp, current_key, ip = _parse_lock_file(fh.readline())
    except IOError:
        time_stamp = 0
        ip = ""
        current_key = ""

    return time_stamp, current_key, ip


def _get_lock_state(lock_time, lock_key, alt_key):

    if not lock_key:
        return LockState.Unlocked
    elif time.time() - lock_time > RESERVATION_TIME:
        return LockState.Unlocked
    elif lock_key == alt_key:
        return LockState.LockedByMe
    else:
        return LockState.LockedByOther


def _validate_lock_key(path, key="", ip="", require_claim=True):

    if not key:
        key = ""

    time_stamp, current_key, lock_ip = _read_lock_file(path)
    lock_state = _get_lock_state(time_stamp, current_key, key)

    if lock_state is LockState.Unlocked and (require_claim or key):
        if not key:
            key = _get_key()
            lock_state = LockState.LockedByMeTemporary
        else:
            lock_state = LockState.LockedByMe
    elif not owns_lock(lock_state) and key == Config().ui_server.master_key:
        lock_state = LockState.LockedByMeTemporary

    if owns_lock(lock_state):
        lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)
        _update_lock(lock_file_path, key, ip)
    elif lock_state is LockState.LockedByOther:
        return lock_state, dict(
            success=False, is_project=True, is_endpoint=True,
            reason="Someone else is working with these results ({0})".format(lock_ip))
    elif require_claim and lock_state is LockState.Unlocked:
        return lock_state, dict(
            success=False, is_project=True, is_endpoint=True,
            reason="Failed to acquire lock though no one was working on project. Please Report")

    return lock_state, dict(is_project=True, **_get_json_lock_response(key, lock_state))


def _discover_projects(path):

    dirs = tuple(chain(*tuple(tuple(os.path.join(root, d) for d in dirs) for root, dirs, _ in os.walk(path))))
    return tuple(d for d in dirs if phenotyper.path_has_saved_project_state(d))


def _get_new_metadata_file_name(project_path, suffix):

    i = 1
    while True:
        path = os.path.join(project_path,
                            Paths().phenotypes_meta_data_original_file_patern.format(i, suffix))
        i += 1
        if not os.path.isfile(path):
            return path


def _get_json_lock_response(lock_key, lock_state):

    if lock_state is LockState.LockedByMeTemporary:
        return {"success": True, "lock_state": lock_state.name}
    else:
        return {"success": owns_lock(lock_state), "lock_state": lock_state.name, "lock_key": lock_key}


def merge_dicts(*dicts):

    return {k: v for k, v in chain(*(d.iteritems() for d in dicts))}


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/results/browse/<path:project>")
    @app.route("/api/results/browse")
    def browse_for_results(project=""):

        local_zone = tz.gettz()

        zone = tz.gettz(request.values.get("time_zone"))
        if not zone:
            zone = tz.gettz()

        path = convert_url_to_path(project)
        is_project = phenotyper.path_has_saved_project_state(path)

        feature_logs = tuple(chain(((
            convert_path_to_url("/api/tools/logs/0/0", c),
            convert_path_to_url("/api/tools/logs/WARNING_ERROR_CRITICAL/0/0", c)) for c in
            glob(os.path.join(path, Paths().phenotypes_extraction_log)))))

        if is_project:
            analysis_date, extraction_date, change_date = phenotyper.get_project_dates(path)
        else:

            return jsonify(**json_response(
                ["feature_logs"],
                dict(
                    success=True, is_project=False, is_endpoint=False,
                    feature_logs=feature_logs,
                    **get_search_results(path, "/api/results/browse"))))

        name = get_project_name(path)

        return jsonify(**json_response(
            ["urls", "add_lock", "remove_lock", "add_meta_data", "meta_data_column_names",
             "phenotype_names", "curves", "quality_index", "gridding", "analysis_instructions", "curve_mark_undo",
             "curve_mark_set", "feature_logs", "export_phenotypes_absolute", "export_phenotypes",
             "phenotype_normalized_names", "has_normalized_data"
             ],
            dict(
                feature_logs=feature_logs,
                project=project,
                is_project=is_project,
                project_name=name,
                add_lock=convert_path_to_url("/api/results/lock/add", path) if is_project else None,
                remove_lock=convert_path_to_url("/api/results/lock/remove", path) if is_project else None,
                add_meta_data=convert_path_to_url("/api/results/meta_data/add", path) if is_project else None,

                meta_data_column_names=convert_path_to_url("/api/results/meta_data/column_names", path)
                if is_project else None,

                curve_mark_undo=convert_path_to_url("/api/results/curve_mark/undo", path) if is_project else None,
                curve_mark_set=convert_path_to_url("/api/results/curve_mark/set", path) if is_project else None,
                phenotype_names=convert_path_to_url("/api/results/phenotype_names", path) if is_project else None,
                phenotype_normalized_names=convert_path_to_url("/api/results/phenotype_normalizable/names", path) if
                is_project else None,
                has_normalized_data=convert_path_to_url("/api/results/has_normalized", path) if is_project else None,
                curves=convert_path_to_url("/api/results/curves", path) if is_project else None,
                quality_index=convert_path_to_url("/api/results/quality_index", path) if is_project else None,
                gridding=convert_path_to_url("/api/results/gridding", path) if is_project else None,

                analysis_date=datetime.fromtimestamp(analysis_date, local_zone).astimezone(zone).isoformat()
                if analysis_date else "",

                extraction_date=datetime.fromtimestamp(extraction_date, local_zone).astimezone(zone).isoformat()
                if extraction_date else "",

                change_date=datetime.fromtimestamp(change_date, local_zone).astimezone(zone).isoformat()
                if change_date else "",

                analysis_instructions=convert_path_to_url("/api/analysis/instructions", path) if is_project else None,

                export_phenotypes=[
                    convert_path_to_url("/api/results/export/phenotypes/{0}".format(norm_state.name), path)
                    for norm_state in phenotyper.NormState] if is_project else None,

                **get_search_results(path, "/api/results/browse"))))

    @app.route("/api/results/lock/add/<path:project>")
    def lock_project(project=""):

        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")

        name = get_project_name(path)
        lock_key = request.values.get("lock_key")
        if not lock_key:
            lock_key = _get_key()

        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)

        response.update({'is_endpoint': True, 'is_project': True, 'project_name': name})

        return jsonify(**response)

    @app.route("/api/results/lock/remove/<path:project>")
    def unlock_project(project=""):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, is_project=False, is_endpoint=True, reason="Not a project")

        name = get_project_name(path)
        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)
        if not owns_lock(lock_state):
            return jsonify(**response)

        response.update({'is_endpoint': True, 'is_project': True, 'project_name': name})
        _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/meta_data/add/<path:project>", methods=["POST"])
    def add_meta_data(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, "/api/results/meta_data/add"))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)
        if not owns_lock(lock_state):
            return jsonify(**response)

        state, _ = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        meta_data_stream = request.files.get('meta_data')
        if not meta_data_stream:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            response["success"] = False
            return jsonify(reason="No file was sent with name/id 'meta_data'", **response)

        file_sufix = request.values.get("file_suffix")
        if not file_sufix:

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            response["success"] = False
            return jsonify(
                reason="The ending of the file (xls, xlsx, ods) was not specified in 'file_suffix'",
                **response)

        meta_data_path = _get_new_metadata_file_name(path, file_sufix)

        try:
            if isinstance(meta_data_stream, FileStorage):
                meta_data_stream.save(meta_data_path)
                meta_data_stream.close()
            else:
                with open(meta_data_path, 'wb') as fh:
                    fh.write(meta_data_stream)
        except IOError:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            response["success"] = False
            return jsonify(reason="Failed to save file, contact server admin.", **response)

        if state.load_meta_data(meta_data_path):
            state.save_state(path, ask_if_overwrite=False)
        else:
            response['success'] = False
            response['reason'] = "Uploaded data doesn't match shapes of the plates"

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/meta_data/column_names/<int:plate>/<path:project>")
    @app.route("/api/results/meta_data/column_names/<path:project>")
    @app.route("/api/results/meta_data/column_names")
    def show_meta_data_headers(plate=None, project=None):

        base_url = "/api/results/meta_data/column_names"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate in state.enumerate_plates]
            response['is_endpoint'] = False
            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        return jsonify(columns=state.meta_data_headers(plate), **response)

    @app.route("/api/results/meta_data/get/<int:plate>/<path:project>")
    @app.route("/api/results/meta_data/get/<path:project>")
    @app.route("/api/results/meta_data/get")
    def get_meta_data(plate=None, project=None):

        base_url = "/api/results/meta_data/get"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)
        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate in state.enumerate_plates]
            response["is_endpoint"] = False
            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if state.meta_data:
            return jsonify(meta_data=state.meta_data[plate].data, **response)
        else:
            response['success'] = False
            response['reason'] = "Project has no meta-data added"
            return jsonify(**response)

    @app.route("/api/results/pinning", defaults={'project': ""})
    @app.route("/api/results/pinning/<path:project>")
    def get_pinning(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/pinning"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, _ = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        pinnings = list(state.plate_shapes)
        return jsonify(pinnings=pinnings, plates=sum(1 for p in pinnings if p is not None), **response)

    @app.route("/api/results/gridding", defaults={'project': ""})
    @app.route("/api/results/gridding/<int:plate>", defaults={'project': ""})
    @app.route("/api/results/gridding/<int:plate>/<path:project>")
    @app.route("/api/results/gridding/<path:project>")
    def get_gridding(project=None, plate=None):

        base_url = "/api/results/gridding"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)
        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate, shape in enumerate(state.enumerate_plates) if shape is not None]

            response['is_endpoint'] = False

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        return send_from_directory(path, Paths().experiment_grid_image_pattern.format(plate + 1))

    @app.route("/api/results/phenotype_names")
    @app.route("/api/results/phenotype_names/<path:project>")
    def get_phenotype_names(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/phenotype_names"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        prepends = ["GenerationTime", "GrowthLag", "ExperimentGrowthYield"]
        phenotypes = sorted(state.phenotype_names())
        phenotypes = [p for p in prepends if p in phenotypes] + [p for p in phenotypes if p not in prepends]

        urls = ["/api/results/phenotype/{0}/{1}".format(phenotype, project)
                for phenotype in phenotypes]

        sort_order = [get_sort_order(p) for p in phenotypes]

        pattern = re.compile(r'([a-z])([A-Z0-9])')

        return jsonify(**json_response(
            ["phenotype_urls"],
            dict(phenotypes=phenotypes,
                 names=[pattern.sub(r'\1 \2', p) for p in phenotypes],
                 phenotype_sort_orders=sort_order,
                 phenotype_urls=urls,
                 **response)))

    @app.route("/api/results/phenotype_normalizable/remove")
    @app.route("/api/results/phenotype_normalizable/remove/<phenotype>/<path:project>")
    def remove_normalizeable_phenotype(project=None, phenotype=""):

        path = convert_url_to_path(project)
        base_url = "/api/results/phenotype_normalizable/remove"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)

        state, name = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if not owns_lock(lock_state):
            return jsonify(**response)

        pheno = PhenotypeDataType.All(phenotype)
        norm_phenos = state.get_normalizable_phenotypes()
        if pheno is None or pheno not in norm_phenos:

            did_supply_phenotype = phenotype == ""
            urls = [base_url + "/{0}/{1}".format(p.name, project) for p in norm_phenos]

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            response['is_endpoint'] = False
            response['success'] = did_supply_phenotype

            return jsonify(**json_response(
                ["urls"],
                {"urls": urls,
                 "reason": "Phenotype not included" if pheno else "Unknown phenotype"},
                **response))

        state.remove_phenotype_from_normalization(pheno)
        state.save_state(path, ask_if_overwrite=False)

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/phenotype_normalizable/add")
    @app.route("/api/results/phenotype_normalizable/add/<phenotype>/<path:project>")
    def add_normalizeable_phenotype(project=None, phenotype=""):

        path = convert_url_to_path(project)
        base_url = "/api/results/phenotype_normalizable/add"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)

        state, name = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if not owns_lock(lock_state):
            return jsonify(**response)

        pheno = PhenotypeDataType.All(phenotype)
        if pheno is None:
            norm_phenos = state.get_normalizable_phenotypes()
            did_supply_phenotype = phenotype == ""
            urls = [base_url + "/{0}/{1}".format(p.name, project) for p in PhenotypeDataType.All()
                    if p not in norm_phenos]

            response['is_endpoint'] = False
            response['success'] = did_supply_phenotype

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            return jsonify(**json_response(
                ["urls"],
                {"urls": urls,
                 "reason": "Unknown phenotype" if did_supply_phenotype else ""},
                **response))

        state.add_phenotype_to_normalization(pheno)
        state.save_state(path, ask_if_overwrite=False)

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/phenotype_normalizable/names")
    @app.route("/api/results/phenotype_normalizable/names/<path:project>")
    def get_normalizeable_phenotype_names(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/phenotype_normalizable/names"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        phenotypes = sorted(state.phenotype_names(normed=True))

        urls = ["/api/results/normalized_phenotype/{0}/{1}".format(phenotype, project)
                for phenotype in phenotypes]
        sort_order = [get_sort_order(p) for p in phenotypes]

        pattern = re.compile(r'([a-z])([A-Z0-9])')

        return jsonify(**json_response(
            ["phenotype_urls"],
            dict(phenotypes=phenotypes,
                 names=[pattern.sub(r'\1 \2', p) for p in phenotypes],
                 phenotype_sort_orders=sort_order,
                 phenotype_urls=urls,
                 **response)))

    @app.route("/api/results/quality_index")
    @app.route("/api/results/quality_index/<int:plate>/<path:project>")
    @app.route("/api/results/quality_index/<path:project>")
    def get_quality_index(project=None, plate=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_project=False,
                           is_endpoint=False,
                           **get_search_results(path, "/api/results/quality_index"))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if plate is None:

            urls = ["/api/results/quality_index/{0}/{1}".format(plate, project)
                    for plate, shape in enumerate(state.enumerate_plates) if shape is not None]

            response['is_endpoint'] = False
            return jsonify(json_response(["urls"], dict(urls=urls, **response)))

        rows, cols = state.get_quality_index(plate)
        return jsonify(dim1_rows=rows.tolist(), dim2_cols=cols.tolist(), **response)

    @app.route("/api/results/normalized_phenotype")
    @app.route("/api/results/normalized_phenotype/<phenotype>/<int:plate>/<path:project>")
    @app.route("/api/results/normalized_phenotype/<int:plate>/<path:project>")
    @app.route("/api/results/normalized_phenotype/<phenotype>/<path:project>")
    def get_normalized_phenotype_data(phenotype=None, project=None, plate=None):

        base_url = "/api/results/normalized_phenotype"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, base_url + "/_NONE_"))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if phenotype is None:

            phenotypes = state.phenotype_names()

            if plate is None:

                urls = [base_url + "/{0}/{1}/{2}".format(phenotype, plate, project)
                        for phenotype, plate in product(phenotypes, state.enumerate_plates)]
            else:

                urls = [base_url + "/{0}/{1}/{2}".format(phenotype, plate, project)
                        for phenotype in phenotypes]

            response['is_endpoint'] = False
            return jsonify(**json_response(["urls"], dict(phenotypes=phenotypes, urls=urls, **response)))

        phenotype_enum = phenotyper.get_phenotype(phenotype)
        is_segmentation_based = state.is_segmentation_based_phenotype(phenotype_enum)

        if plate is None:

            urls = []
            plate_indices = []
            for plate, shape in enumerate(state.plate_shapes):
                if shape is not None:
                    urls.append(base_url + "/{0}/{1}/{2}".format(phenotype, plate, project))
                    plate_indices.append(plate)

            response['is_endpoint'] = False
            return jsonify(**json_response(
                ["urls"],
                dict(
                    urls=urls, plate_indices=plate_indices, is_segmentation_based=is_segmentation_based, **response)))

        try:
            plate_data = state.get_phenotype(phenotype_enum, normalized=True)[plate]
        except ValueError:
            response['success'] = False
            return jsonify(reason="Phenotype hasn't been normalized", plate=plate, phenotype=phenotype, **response)

        if plate_data is None:
            response['success'] = False
            return jsonify(reason="Phenotype hasn't been normalized", plate=plate, phenotype=phenotype, **response)

        qindex_rows, qindex_cols = state.get_quality_index(plate)

        return jsonify(
            data=plate_data.tojson(), plate=plate, phenotype=phenotype,
            is_segmentation_based=is_segmentation_based,
            qindex_rows=qindex_rows.tolist(),
            qindex_cols=qindex_cols.tolist(),
            **merge_dicts(
                {filt.name: tuple(v.tolist() for v in plate_data.where_mask_layer(filt))
                 for filt in Filter if filt != Filter.OK},
                response))

    @app.route("/api/results/phenotype")
    @app.route("/api/results/phenotype/<phenotype>/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<phenotype>/<path:project>")
    def get_phenotype_data(phenotype=None, project=None, plate=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, "/api/results/phenotype/_NONE_"))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if phenotype is None:

            phenotypes = state.phenotype_names()
            response['is_endpoint'] = False
            if plate is None:

                urls = ["/api/results/phenotype/{0}/{1}/{2}".format(phenotype, plate, project)
                        for phenotype, plate in product(phenotypes, state.enumerate_plates)]
            else:

                urls = ["/api/results/phenotype/{0}/{1}/{2}".format(phenotype, plate, project)
                        for phenotype in phenotypes]

            return jsonify(**json_response(["urls"], dict(phenotypes=phenotypes, urls=urls, **response)))

        phenotype_enum = phenotyper.get_phenotype(phenotype)
        is_segmentation_based = state.is_segmentation_based_phenotype(phenotype_enum)

        if plate is None:

            urls = []
            plate_indices = []
            for plate, shape in enumerate(state.plate_shapes):
                if shape is not None:
                    urls.append("/api/results/phenotype/{0}/{1}/{2}".format(phenotype, plate, project))
                    plate_indices.append(plate)
            response['is_endpoint'] = False
            return jsonify(**json_response(
                ["urls"],
                dict(
                    urls=urls, plate_indices=plate_indices, is_segmentation_based=is_segmentation_based, **response)))

        try:
            plate_data = state.get_phenotype(phenotype_enum)[plate]
        except ValueError:
            response['success'] = False
            return jsonify(
                reason="Phenotype hasn't been extracted",
                plate=plate, phenotype=phenotype, **response)

        if plate_data is None:
            response['success'] = False
            return jsonify(
                reason="Phenotype hasn't been extracted",
                plate=plate, phenotype=phenotype, **response)

        qindex_rows, qindex_cols = state.get_quality_index(plate)

        return jsonify(
            data=plate_data.tojson(), plate=plate, phenotype=phenotype,
            qindex_rows=qindex_rows.tolist(),
            qindex_cols=qindex_cols.tolist(),
            is_segmentation_based=is_segmentation_based,
            **merge_dicts(
                {filt.name: tuple(v.tolist() for v in plate_data.where_mask_layer(filt))
                 for filt in Filter if filt != Filter.OK},
                response))

    @app.route("/api/results/curve_mark/names")
    def curve_mark_names():
        """Get the names

        Returns: json-objects

        """
        return jsonify(**{Filter.Empty.name: {'text': "Empty",
                                              'value': Filter.Empty.value,
                                              'user_settable': True},
                          Filter.BadData.name: {'text': "Bad Data",
                                                'value': Filter.BadData.value,
                                                'user_settable': True},
                          Filter.NoGrowth.name: {'text': "No Growth",
                                                 'value': Filter.NoGrowth.value,
                                                 'user_settable': True},
                          Filter.OK.name: {'text': "OK",
                                           'value': Filter.OK.value,
                                           'user_settable': True},
                          Filter.UndecidedProblem.name: {'text': "Undecided Problem",
                                                         'value': Filter.UndecidedProblem.value,
                                                         'user_settable': False}})

    @app.route("/api/results/curve_mark/undo")
    @app.route("/api/results/curve_mark/undo/<int:plate>/<path:project>")
    @app.route("/api/results/curve_mark/undo/<path:project>")
    def undo_curve_mark(plate=None, project=None):
        """Undo last log2_curve mark

        Args:
            plate: int, index of plate to invoke undo on
            project: the url-formatted path

        Returns: json-object
            Specific keys:
            'had_effect' key implies if undo did anything, typically
                False means there's no more undo.

        """
        url_root = "/api/results/curve_mark/undo"
        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)

        state, name = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if not owns_lock(lock_state):
            return jsonify(**response)

        # If plate not submitted give plate completing paths
        if plate is None:

            response['is_endpoint'] = True

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        had_effect = state.undo(plate)
        state.save_state(path, ask_if_overwrite=False)

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(had_effect=had_effect, **response)

    @app.route("/api/results/curve_mark/set")
    @app.route("/api/results/curve_mark/set/<mark>/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>",
               methods=["POST", "GET"])
    @app.route("/api/results/curve_mark/set/<mark>/<int:plate>/<path:project>", methods=["POST", "GET"])
    @app.route("/api/results/curve_mark/set/<mark>/<phenotype>/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>")
    @app.route("/api/results/curve_mark/set/<mark>/<phenotype>/<int:plate>/<path:project>", methods=["POST", "GET"])
    @app.route("/api/results/curve_mark/set/<path:project>")
    def set_curve_mark( mark=None, phenotype=None, plate=None, d1_row=None, d2_col=None, project=None):
        """Sets a log2_curve filter mark for a position or list of positions

        If several positions should be marked at once the `d1_row` and
        `d2_col` should be omitted from the url and instead be
        submitted via POST.

        Supports submitting `d1_row`, `d2_col` and `phenotype` via GET
        and POST as well as in the actual url.

        Args:
            plate: int, plate index
            d1_row: int or tuple of ints, the outer coordinate(s) of
                position(s) to be marked.
            d2_col: int or tuple of ints, the inner coordinate(s) of
                position(s) to be marked.
            phenotype: str, name of the phenotype to mark. If omitted
                and no `phenotype` value submitted by POST, mark is
                applied to all phenotypes.
            mark: str, name of the marking to make.
            project: str, url-formatted path to the project.

        Returns: json-object

        """
        url_root = "/api/results/curve_mark/set"
        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)

        state, _ = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if not owns_lock(lock_state):
            return jsonify(**response)

        # If mark is lacking create urls.
        if mark is None:

            response['is_endpoint'] = False

            urls = ["{0}/{1}/{2}/{3}".format(url_root, m.name, i, project)
                    for (i, p), m in product(enumerate(state.plate_shapes), phenotyper.Filter) if p is not None and
                    m is not Filter.UndecidedProblem]

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        elif mark == Filter.UndecidedProblem.name:

            response['success'] = False

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            return jsonify(reason="User not allowed to set mark {0}".format(mark), **response)

        # Update phenotype if submitted via POST
        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if phenotype is None:
            phenotype = data_object.get("phenotype", default=None)

        # If plate not submitted give plate completing paths
        if plate is None:

            urls = ["{0}/{1}/{2}/{3}".format(url_root, "/".join((mark, phenotype) if phenotype else mark), i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            response['is_endpoint'] = False

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        # Process position(s) info
        if d1_row is None:
            d1_row = data_object.get("d1_row", default=None)

        if d2_col is None:
            d1_row = data_object.get("d2_col",  default=None)

        # Ensure format will be correctly interpreted by numpy
        try:
            outer = d1_row if isinstance(d1_row, int) else tuple(d1_row)
        except TypeError:
            outer = None
        try:
            inner = d2_col if isinstance(d2_col, int) else tuple(d2_col)
        except TypeError:
            inner = None

        if outer is None or inner is None:

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            response['success'] = False

            return jsonify(
                reason="Positional coordinates are not valid ({0}, {1})".format(outer, inner),
                **response)

        # Validate that the mark is understood
        try:
            mark = phenotyper.Filter[mark]
        except KeyError:

            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            response['success'] = False

            return jsonify(
                reason="Invalid position mark ({0}), supported {1}".format(mark, tuple(f for f in phenotyper.Filter)),
                **response)

        # Validate that the phenotype is understood and exists
        if phenotype is not None:

            phenotype = infer_phenotype_from_name(phenotype)
            if phenotype not in state:

                if lock_state is LockState.LockedByMeTemporary:
                    _remove_lock(path)

                response['success'] = False

                return jsonify(
                    reason="Phenotype '{0}' not included in extraction".format(phenotype,
                                                                               tuple(f for f in phenotyper.Filter)),
                    **response)

        if not state.add_position_mark(plate, (outer, inner), phenotype, mark):
            response['success'] = False

            return jsonify(
                reason="Setting mark refused, probably trying to set NoGrowth or Empty for individual phenotype.",
                **response)

        state.save_state(path, ask_if_overwrite=False)

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/curves")
    @app.route("/api/results/curves/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>")
    @app.route("/api/results/curves/<int:plate>/<path:project>")
    @app.route("/api/results/curves/<path:project>")
    def get_growth_data(plate=None, d1_row=None, d2_col=None, project=None):

        url_root = "/api/results/curves"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]
            response['is_endpoint'] = False
            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if d1_row is None or d2_col is None:
            shape = tuple(state.plate_shapes)[plate]
            if shape is None:
                response['success'] = False
                return jsonify(reason="Plate not included in project", **response)

            response['is_endpoint'] = False

            urls = ["{0}/{1}/{2}/{3}/{4}".format(url_root, plate, d1, d2, project) for d1, d2 in
                    product(range(shape[0]) if d1_row is None else [d1_row],
                            range(shape[1]) if d2_col is None else [d2_col])]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        segmentations = state.get_curve_phases(plate, d1_row, d2_col)
        if segmentations is not None:
            segmentations = segmentations.tolist()

        film_url = ["/api/results/movie/make/{0}/{1}/{2}/{3}".format(plate, d1_row, d2_col, project)]
        colony_image = ["/api/compile/colony_image/{4}/{0}/{1}/{2}/{3}".format(plate, d1_row, d2_col, project, t)
                        for t, _ in enumerate(state.times)]

        mark_all_urls = [
            "/api/results/curve_mark/set/{0}/{1}/{2}/{3}/{4}".format(m.name, plate, d1_row, d2_col, project)
            for m in phenotyper.Filter]

        return jsonify(time_data=state.times.tolist(),
                       smooth_data=state.smooth_growth_data[plate][d1_row, d2_col].tolist(),
                       raw_data=state.raw_growth_data[plate][d1_row, d2_col].tolist(),
                       segmentations=segmentations,
                       **json_response(["film_urls", "colony_image", "mark_all_urls"],
                                       dict(film_urls=film_url, colony_image=colony_image,
                                            mark_all_urls=mark_all_urls, **response)))

    @app.route("/api/results/movie/make")
    @app.route("/api/results/movie/make/<film_type>/<int:plate>/<int:outer_dim>/<int:inner_dim>/<path:project>")
    @app.route("/api/results/movie/make/<int:plate>/<int:outer_dim>/<int:inner_dim>/<path:project>")
    @app.route("/api/results/movie/make/<int:plate>/<path:project>")
    @app.route("/api/results/movie/make/<path:project>")
    def get_film(project=None, film_type=None, plate=None, outer_dim=None, inner_dim=None):

        url_root = "/api/results/movie/make"

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)
        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            response['is_endpoint'] = False

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if outer_dim is None or inner_dim is None:

            shape = tuple(state.plate_shapes)[plate]
            if shape is None:
                response['success'] = False
                return jsonify(reason="Plate not included in project", **response)

            urls = ["{0}/{1}/{2}/{3}/{4}".format(url_root, plate, d1, d2, project) for d1, d2 in
                    product(range(shape[0]) if outer_dim is None else [outer_dim],
                            range(shape[1]) if inner_dim is None else [inner_dim])]

            response['is_endpoint'] = False

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if film_type is None or film_type not in FILM_TYPES:

            urls = ["{0}/{1}/{2}/{3}/{4}/{5}".format(url_root, f_type, plate, outer_dim, inner_dim, project) for
                    f_type in FILM_TYPES]

            response['is_endpoint'] = False

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        save_path = os.path.join(path, "qc_film_{0}_{1}_{2}.{3}.avi".format(plate, outer_dim, inner_dim, film_type))
        if _make_film(film_type, save_target=save_path, pos=(plate, outer_dim, inner_dim), path=path):

            return send_from_directory(path, os.path.basename(save_path))
        else:
            response['success'] = False
            return jsonify(reason="Error while producing film", **response)

    @app.route("/api/results/normalize")
    @app.route("/api/results/normalize/<path:project>")
    def _do_normalize(project):
        """Preform normalization

        Arga:

            project: str, url-formatted path to the project.

        """

        url_root = "/api/results/normalize"

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        name = get_project_name(path)
        lock_state, response = _validate_lock_key(
            path, request.values.get("lock_key"), request.remote_addr, require_claim=True)

        response.update({'is_endpoint': True, 'is_project': True, 'project_name': name})

        state, name = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if not owns_lock(lock_state):
            return jsonify(**response)

        state.normalize_phenotypes()
        state.save_state(path, ask_if_overwrite=False)

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/normalize/reference/offsets")
    def _get_offset_names():

        names = [o.name for o in phenotyper.Offsets]
        values = [o.value for o in phenotyper.Offsets]
        return jsonify(success=True, offset_names=names, offset_values=values)

    @app.route("/api/results/normalize/reference/set/<int:plate>/<offset>/<path:project>")
    @app.route("/api/results/normalize/reference/set/<offset>/<path:project>")
    def _set_normalization_offset(project, offset, plate=None):
        """Sets a normalization offset

        Arga:

            project: str, url-formatted path to the project.
            offset: One of the four offset names
            plate: Optional, if supplied only apply to a
                specific plate, else apply to all plates.

        """
        url_root = "/api/results/normalize/reference/set"

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(
            path, lock_key, request.remote_addr, require_claim=True)

        state, _ = _get_state_update_response(path, response)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        if not owns_lock(lock_state):
            return jsonify(**response)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, plate, project)
                    for plate in state.enumerate_plates]
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            response['is_endpoint'] = False

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        try:
            offset = Offsets[offset]
        except ValueError:
            urls = [url_root + "/{0}/{1}/{2}/{3}".format(url_root, plate, off.name, project) for off in Offsets]
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)

            response['success'] = False

            return jsonify(reason="Bad offset name {0}".format(offset),
                           **json_response(["urls"], dict(urls=urls, **response)))

        state.set_control_surface_offsets(offset, plate)
        state.save_state(path, ask_if_overwrite=False)

        if lock_state is LockState.LockedByMeTemporary:
            _remove_lock(path)

        return jsonify(**response)

    @app.route("/api/results/normalize/reference/get/<int:plate>/<path:project>")
    def _get_normalization_offset(project, plate):
        """Gets the normalization offset of a plate

        Arga:

            project: str, url-formatted path to the project.
            plate: The plate in question

        """
        url_root = "/api/results/normalize/reference/get"

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        offset = state.get_control_surface_offset(plate)

        try:
            offset = infer_offset(offset)
        except ValueError:
            response['success'] = False
            return jsonify(reason="Non standard offset used by someone hacking the project", **response)

        return jsonify(offset_name=offset.name, offset_value=offset.value,
                       offset_pattern=offset().tolist(), **response)

    @app.route("/api/results/has_normalized/<path:project>")
    def _get_has_been_normed(project):
        """If the project has normalized data.

        Arga:

            project: str, url-formatted path to the project.


        """
        url_root = "/api/results/has_normalized"

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        return jsonify(has_normalized=state.has_normalized_data, **response)

    @app.route("/api/results/export/phenotypes/<save_data>/<path:project>")
    def export_phenotypes(project, save_data=""):
        url_root = "/api/results/export/phenotypes"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(
                path, "{0}/{1}".format(url_root, save_data)))))

        lock_key = request.values.get("lock_key")
        lock_state, response = _validate_lock_key(path, lock_key, request.remote_addr, require_claim=False)

        state, name = _get_state_update_response(path, response, success=True)

        if state is None:
            if lock_state is LockState.LockedByMeTemporary:
                _remove_lock(path)
            return jsonify(**response)

        try:
            save_data = phenotyper.NormState[save_data]
        except KeyError:
            return jsonify(**json_response(
                ['urls'],
                dict(
                    urls=["{0}/{1}/{2}".format(url_root, sd.name, project) for sd in
                          phenotyper.NormState],
                    reason="SaveData-type {0} not known".format(save_data)
                ),
                success=False))

        if state.save_phenotypes(
                dir_path=path,
                save_data=save_data,
                ask_if_overwrite=False):

            return serve_zip_file(
                "{0}.phenotypes.{1}.zip".format(name, save_data.name),
                *(state.get_csv_file_name(path, save_data, plate) for plate in state.enumerate_plates))

        else:
            return jsonify(**json_response(
                [],
                dict(
                    reason="Saving phenotypes failed for some unknown reason"
                ),
                success=False))

    # End of UI extension with qc-functionality
    return True
