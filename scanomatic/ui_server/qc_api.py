import os
import time
from datetime import datetime
from dateutil import tz
from flask import request, Flask, jsonify, send_from_directory
from itertools import chain, product
from subprocess import call
import uuid

from scanomatic.data_processing import phenotyper
from scanomatic.data_processing.phenotypes import get_sort_order
from scanomatic.generics.phenotype_filter import Filter
from scanomatic.io.paths import Paths
from scanomatic.io.app_config import Config
from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url, get_search_results, \
    get_project_name, json_response

RESERVATION_TIME = 60 * 5
FILM_TYPES = {'colony': 'animate_colony_growth("{save_target}", {pos}, "{path}")',
              'detection': 'animate_blob_detection("{save_target}", {pos}, "{path}")',
              '3d': 'animate_3d_colony("{save_target}", {pos}, "{path}")'}


def _make_film(film_type, save_target=None, pos=None, path=None):
    code = FILM_TYPES[film_type].format(save_target=save_target, pos=pos, path=path)
    retcode = call(['python', '-c', 'from scanomatic.qc import analysis_results;analysis_results.{0}'.format(code)])
    return retcode == 0


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


def _read_lock_file(path):

    def parse(data):

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

    lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)
    try:
        with open(lock_file_path, 'r') as fh:
            time_stamp, current_key, ip = parse(fh.readline())
    except IOError:
        time_stamp = 0
        ip = ""
        current_key = ""

    return time_stamp, current_key, ip


def _validate_lock_key(path, key="", ip=""):

    if not key:
        key = ""

    time_stamp, current_key, lock_ip = _read_lock_file(path)
    locked_by_other = False
    if not(key == current_key or key == Config().ui_server.master_key or not current_key or
           time.time() - time_stamp > RESERVATION_TIME):
        locked_by_other = True
    else:
        current_key = ""
        lock_ip = ""

    if locked_by_other:
        return locked_by_other, "", lock_ip

    if key:
        lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)
        locked_by_me = _update_lock(lock_file_path, key, ip)
        return locked_by_me, key, ip
    else:
        return locked_by_other, current_key, lock_ip


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


def _get_json_lock_response(lock_key, data=None):

    if data is None:
        data = dict()

    data["read_only"] = not lock_key
    data["lock_key"] = lock_key
    return data


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

        if is_project:
            analysis_date, extraction_date, change_date = phenotyper.get_project_dates(path)
        else:

            return jsonify(success=True, is_project=False, is_endpoint=False,
                           **get_search_results(path, "/api/results/browse"))

        name = get_project_name(path)

        return jsonify(**json_response(
            ["urls", "add_lock", "remove_lock", "add_meta_data", "meta_data_column_names",
             "phenotype_names", "curves", "quality_index", "gridding", "analysis_instructions", "curve_mark_undo",
             "curve_mark_set"],
            dict(
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
                **get_search_results(path, "/api/results/browse"))))

    @app.route("/api/results/lock/add/<path:project>")
    def lock_project(project=""):

        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")
        locked, key, ip = _validate_lock_key(path, _get_key(), request.remote_addr)
        name = get_project_name(path)

        if key and locked:
            return jsonify(success=True, is_project=True, is_endpoint=True, lock_key=key, project_name=name)
        elif locked:
            return jsonify(success=False, is_project=True, is_endpoint=True, project_name=name,
                           reason="Someone else ({0}) is working with these results".format(ip))
        else:
            return jsonify(success=False, is_project=True, is_endpoint=True, project_name=name,
                           reason="No one is working on it but locking was refused, should not happen. Please report")

    @app.route("/api/results/lock/remove/<path:project>")
    def unlock_project(project=""):

        path = convert_url_to_path(project)

        locked, key, ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, is_project=False, is_endpoint=True, reason="Not a project")

        if locked and not key:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Invalid key, locked by {0}".format(ip))
        elif not locked:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Failed to acquire lock though no one was working on project. Please Report")

        _remove_lock(path)
        name = get_project_name(path)
        return jsonify(success=True, is_project=True, is_endpoint=True, project_name=name)

    @app.route("/api/results/meta_data/add/<path:project>", methods=["POST"])
    def add_meta_data(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, "/api/results/meta_data/add"))))

        name = get_project_name(path)
        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        if locked and not lock_key:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           project_name=name,
                           reason="Someone else is working with these results ({0})".format(lock_ip))
        elif not locked:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Failed to acquire lock though no one was working on project. Please Report")

        meta_data_stream = request.files.get('meta_data')
        if not meta_data_stream:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           project_name=name,
                           reason="No file was sent with name/id 'meta_data'")

        file_sufix = request.values.get("file_suffix")
        if not file_sufix:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           project_name=name,
                           reason="The ending of the file (xls, xlsx, ods) was not specified in 'file_suffix'")

        meta_data_path = _get_new_metadata_file_name(path, file_sufix)
        try:
            with open(meta_data_path, 'wb') as fh:
                fh.write(meta_data_stream)
        except IOError:
            return jsonify(success=False, reason="Failed to save file, contact server admin.",
                           is_endpoint=True, is_project=True, project_name=name, **_get_json_lock_response(lock_key))

        state = phenotyper.Phenotyper.LoadFromState(path)
        state.load_meta_data(meta_data_path)
        state.save_state(path, ask_if_overwrite=False)

        return jsonify(success=True, is_project=True, is_endpoint=True, project_name=name,
                       **_get_json_lock_response(lock_key))

    @app.route("/api/results/meta_data/column_names/<int:plate>/<path:project>")
    @app.route("/api/results/meta_data/column_names/<path:project>")
    @app.route("/api/results/meta_data/column_names")
    def show_meta_data_headers(plate=None, project=None):

        base_url = "/api/results/meta_data/column_names"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))
        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate in state.enumerate_plates]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        return jsonify(success=True, is_endpoint=True, columns=state.meta_data_headers(plate), **response)

    @app.route("/api/results/meta_data/get/<int:plate>/<path:project>")
    @app.route("/api/results/meta_data/get/<path:project>")
    @app.route("/api/results/meta_data/get")
    def get_meta_data(plate=None, project=None):

        base_url = "/api/results/meta_data/get"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))
        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate in state.enumerate_plates]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if state.meta_data:
            return jsonify(success=True, is_endpoint=True, meta_data=state.meta_data[plate].data, **response)
        else:
            return jsonify(success=False, reason="Project has no meta-data added", is_endpoint=True, **response)

    @app.route("/api/results/pinning", defaults={'project': ""})
    @app.route("/api/results/pinning/<path:project>")
    def get_pinning(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/pinning"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        pinnings = list(state.plate_shapes)
        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        return jsonify(success=True, is_project=True, is_endpoint=True, project_name=get_project_name(path),
                       pinnings=pinnings, plates=sum(1 for p in pinnings if p is not None),
                       **_get_json_lock_response(lock_key))

    @app.route("/api/results/gridding", defaults={'project': ""})
    @app.route("/api/results/gridding/<int:plate>", defaults={'project': ""})
    @app.route("/api/results/gridding/<int:plate>/<path:project>")
    @app.route("/api/results/gridding/<path:project>")
    def get_gridding(project=None, plate=None):

        base_url = "/api/results/gridding"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        _, lock_key, _ = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate, shape in enumerate(state.enumerate_plates) if shape is not None]

            return jsonify(**json_response(
                ["urls"], dict(is_project=True, urls=urls, project_name=name, **_get_json_lock_response(lock_key))))

        return send_from_directory(path, Paths().experiment_grid_image_pattern.format(plate + 1))

    @app.route("/api/results/phenotype_names")
    @app.route("/api/results/phenotype_names/<path:project>")
    def get_phenotype_names(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/phenotype_names"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(jsonify(is_project=False, **get_search_results(path, base_url)))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        _, lock_key, _ = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        urls = ["/api/results/phenotype/{0}/{1}".format(phenotype, project)
                for phenotype in state.phenotype_names()]

        sort_order = [get_sort_order(p) for p in state.phenotype_names()]

        return jsonify(**json_response(
            ["phenotype_urls"],
            dict(phenotypes=state.phenotype_names(),
                 phenotype_sort_orders=sort_order,
                 is_project=True,
                 phenotype_urls=urls,
                 project_name=name,
                 **_get_json_lock_response(lock_key))))

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

        state = phenotyper.Phenotyper.LoadFromState(path)
        _, lock_key, _ = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))
        if plate is None:

            urls = ["/api/results/quality_index/{0}/{1}".format(plate, project)
                    for plate, shape in enumerate(state.enumerate_plates) if shape is not None]

            return jsonify(json_response(["urls"], dict(urls=urls, **response)))

        rows, cols = state.get_quality_index(plate)
        return jsonify(success=True, is_endpoint=True, dim1_rows=rows.tolist(), dim2_cols=cols.tolist(), **response)

    @app.route("/api/results/phenotype")
    @app.route("/api/results/phenotype/<phenotype>/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<phenotype>/<path:project>")
    def get_phenotype_data(phenotype=None, project=None, plate=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, "/api/results/phenotype/_NONE_"))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        _, lock_key, _ = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))
        if phenotype is None:

            phenotypes = state.phenotype_names()

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

            return jsonify(**json_response(
                ["urls"],
                dict(
                    urls=urls, plate_indices=plate_indices, is_segmentation_based=is_segmentation_based, **response)))

        plate_data = state.get_phenotype(phenotype_enum)[plate]

        return jsonify(
            success=True, data=plate_data.tojson(), plate=plate, phenotype=phenotype, is_endpoint=True,
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
        """Undo last curve mark

        Args:
            plate: int, index of plate to invoke undo on
            project: the url-formatted path

        Returns: json-object
            Specific keys:
            'had_effect' key implies if undo did anything, typically
                False means there's no more undo.

        """
        url_root = "/api/results/set_curve_mark/undo"
        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))

        # Validate lock, without lock nothing will happen
        if locked and not lock_key:
            return jsonify(success=False, reason="Failed to acquire lock on project (owned by {0})".format(lock_ip),
                           is_endpoint=True, **response)
        elif not locked:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Failed to acquire lock though no one was working on project. Please Report")

        state = phenotyper.Phenotyper.LoadFromState(path)

        # If plate not submitted give plate completing paths
        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        had_effect = state.undo(plate)
        state.save_state(path, ask_if_overwrite=False)

        return jsonify(success=True, is_endpoint=True, had_effect=had_effect, **response)

    @app.route("/api/results/curve_mark/set")
    @app.route("/api/results/curve_mark/set/<mark>/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>",
               methods=["POST", "GET"])
    @app.route("/api/results/curve_mark/set/<mark>/<int:plate>/<path:project>", methods=["POST", "GET"])
    @app.route("/api/results/curve_mark/set/<mark>/<phenotype>/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>")
    @app.route("/api/results/curve_mark/set/<mark>/<phenotype>/<int:plate>/<path:project>", methods=["POST", "GET"])
    @app.route("/api/results/curve_mark/set/<path:project>")
    def set_curve_mark(plate=None, d1_row=None, d2_col=None, phenotype=None, mark=None, project=None):
        """Sets a curve filter mark for a position or list of positions

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

        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))

        # Validate lock, without lock nothing will happen
        if locked and not lock_key:
            return jsonify(success=False, reason="Failed to acquire lock on project (owned by {0})".format(lock_ip),
                           is_endpoint=True, **response)
        elif not locked:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Failed to acquire lock though no one was working on project. Please Report",
                           **response)

        state = phenotyper.Phenotyper.LoadFromState(path)

        # If mark is lacking create urls.
        if mark is None:

            urls = ["{0}/{1}/{2}/{3}".format(url_root, m.name, i, project)
                    for (i, p), m in product(enumerate(state.plate_shapes), phenotyper.Filter) if p is not None]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        # Update phenotype if submitted via POST
        if phenotype is None:
            phenotype = request.values.get("phenotype", default=None)

        # If plate not submitted give plate completing paths
        if plate is None:

            urls = ["{0}/{1}/{2}/{3}".format(url_root, "/".join((mark, phenotype) if phenotype else mark), i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        # Process position(s) info
        if d1_row is None:
            d1_row = request.values.get("d1_row", default=None)

        if d2_col is None:
            d1_row = request.values.get("d2_col",  default=None)

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
            return jsonify(
                success=False, reason="Positional coordinates are not valid ({0}, {1})".format(outer, inner),
                is_endpoint=True, **response)

        # Validate that the mark is understood
        try:
            mark = phenotyper.Filter[mark]
        except KeyError:
            return jsonify(
                success=False,
                reason="Invalid position mark ({0}), supported {1}".format(mark, tuple(f for f in phenotyper.Filter)),
                is_endpoint=True, **response)

        # Validate that the phenotype is understood and exists
        if phenotype is not None and phenotype not in state:
            return jsonify(
                success=False,
                reason="Phenotype '{0}' not included in extraction".format(mark, tuple(f for f in phenotyper.Filter)),
                is_endpoint=True, **response
            )

        state.add_position_mark(plate, (outer, inner), phenotype, mark)
        state.save_state(path, ask_if_overwrite=False)

        return jsonify(success=True, is_endpoint=True, **response)

    @app.route("/api/results/curves")
    @app.route("/api/results/curves/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>")
    @app.route("/api/results/curves/<int:plate>/<path:project>")
    @app.route("/api/results/curves/<path:project>")
    def get_growth_data(plate=None, d1_row=None, d2_col=None, project=None):

        url_root = "/api/results/curves"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        locked, lock_key, lock_ip = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if d1_row is None or d2_col is None:

            shape = tuple(state.plate_shapes)[plate]
            if shape is None:
                return jsonify(success=False, reason="Plate not included in project")

            urls = ["{0}/{1}/{2}/{3}/{4}".format(url_root, plate, d1, d2, project) for d1, d2 in
                    product(range(shape[0]) if d1_row is None else [d1_row],
                            range(shape[1]) if d2_col is None else [d2_col])]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        segmentations = state.get_curve_segments(plate, d1_row, d2_col)
        if segmentations is not None:
            segmentations = segmentations.tolist()

        film_url = ["/api/results/movie/make/{0}/{1}/{2}/{3}".format(plate, d1_row, d2_col, project)]
        colony_image = ["/api/compile/colony_image/{4}/{0}/{1}/{2}/{3}".format(plate, d1_row, d2_col, project, t)
                        for t, _ in enumerate(state.times)]

        mark_all_urls = ["/api/results/mark/set/{0}/{1}/{2}/{3}/{4}".format(m.name, plate, d1_row, d2_col, project)
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

        state = phenotyper.Phenotyper.LoadFromState(path)
        _, lock_key, _ = _validate_lock_key(path, request.values.get("lock_key"), request.remote_addr)
        name = get_project_name(path)
        response = dict(is_project=True, project_name=name, **_get_json_lock_response(lock_key))

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if outer_dim is None or inner_dim is None:

            shape = tuple(state.plate_shapes)[plate]
            if shape is None:
                return jsonify(success=False, reason="Plate not included in project")

            urls = ["{0}/{1}/{2}/{3}/{4}".format(url_root, plate, d1, d2, project) for d1, d2 in
                    product(range(shape[0]) if outer_dim is None else [outer_dim],
                            range(shape[1]) if inner_dim is None else [inner_dim])]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        if film_type is None or film_type not in FILM_TYPES:

            urls = ["{0}/{1}/{2}/{3}/{4}/{5}".format(url_root, f_type, plate, outer_dim, inner_dim, project) for
                    f_type in FILM_TYPES]

            return jsonify(**json_response(["urls"], dict(urls=urls, **response)))

        save_path = os.path.join(path, "qc_film_{0}_{1}_{2}.{3}.avi".format(plate, outer_dim, inner_dim, film_type))
        if _make_film(film_type, save_target=save_path, pos=(plate, outer_dim, inner_dim), path=path):

            return send_from_directory(path, os.path.basename(save_path))
        else:
            return jsonify(success=False, reason="Error while producing film")

    # End of UI extension with qc-functionality
    return True
