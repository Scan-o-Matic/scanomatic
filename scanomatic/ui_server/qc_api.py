import os
import time
from datetime import datetime
from dateutil import tz
from flask import request, Flask, jsonify, send_from_directory
from itertools import chain, product

import uuid

from scanomatic.data_processing import phenotyper
from scanomatic.data_processing.phenotypes import get_sort_order
from scanomatic.generics.phenotype_filter import Filter
from scanomatic.io.paths import Paths
from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url, get_search_results, \
    get_project_name, json_response
RESERVATION_TIME = 60 * 5


def _add_lock(path):

    key = uuid.uuid4().hex
    _update_lock(path, key)
    return key


def _update_lock(lock_file_path, key):

    with open(lock_file_path, 'w') as fh:
        fh.write("|".join((str(time.time()), str(key))))
    return True


def _remove_lock(path):

    lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)
    os.remove(lock_file_path)

    return True


def _validate_lock_key(path, key=""):

    if not key:
        key = ""

    lock_file_path = os.path.join(path, Paths().ui_server_phenotype_state_lock)

    locked = False
    try:
        with open(lock_file_path, 'r') as fh:
            time_stamp, current_key = fh.readline().split("|")
            time_stamp = float(time_stamp)
            if not(key == current_key or time.time() - time_stamp > RESERVATION_TIME):
                locked = True
    except IOError:
        pass
    except ValueError:
        pass

    if locked:
        return ""

    if key:
        _update_lock(lock_file_path, key)
        return key
    else:
        return _add_lock(lock_file_path)


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


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/results/browse/<path:project>")
    @app.route("/api/results/browse")
    @app.route("/api/results/browse/")
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
            ["urls"],
            dict(
                project=project,
                is_project=is_project,
                project_name=name,
                add_lock=convert_path_to_url("/api/results/lock/add", path) if is_project else None,
                remove_lock=convert_path_to_url("/api/results/lock/remove", path) if is_project else None,
                add_meta_data=convert_path_to_url("/api/results/meta_data/add", path) if is_project else None,

                meta_data_column_names=convert_path_to_url("/api/results/meta_data/column_names", path)
                if is_project else None,

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
        key = _validate_lock_key(path, "")
        name = get_project_name(path)

        if key:
            return jsonify(success=True, is_project=True, is_endpoint=True, lock_key=key, project_name=name)
        else:
            return jsonify(success=False, is_project=True, is_endpoint=True, project_name=name,
                           reason="Someone else is working with these results")

    @app.route("/api/results/lock/remove/<path:project>")
    def unlock_project(project=""):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, is_project=False, is_endpoint=True, reason="Not a project")

        if not _validate_lock_key(path, request.values.get("lock_key")):
            return jsonify(success=False, is_project=True, is_endpoint=True, reason="Invalid key")

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
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        if not lock_key:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           project_name=name,
                           reason="Someone else is working with these results")

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
                           read_only=not lock_key, lock_key=lock_key,
                           is_endpoint=True, is_project=True, project_name=name)

        state = phenotyper.Phenotyper.LoadFromState(path)
        state.load_meta_data(meta_data_path)
        state.save_state(path, ask_if_overwrite=False)

        return jsonify(success=True, is_project=True, is_endpoint=True, project_name=name,
                       read_only=not lock_key, lock_key=lock_key)

    @app.route("/api/results/meta_data/column_names/<int:plate>/<path:project>")
    @app.route("/api/results/meta_data/column_names/<path:project>")
    @app.route("/api/results/meta_data/column_names")
    @app.route("/api/results/meta_data/column_names/")
    def show_meta_data_headers(plate=None, project=None):

        base_url = "/api/results/meta_data/column_names"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = get_project_name(path)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate in state.enumerate_plates]

            return jsonify(**json_response(
                ["urls"],
                dict(read_only=not lock_key, lock_key=lock_key, is_project=True, project_name=name, urls=urls)))

        return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                       is_project=True, is_endpoint=True, project_name=name,
                       columns=state.meta_data_headers(plate))

    @app.route("/api/results/meta_data/get/<int:plate>/<path:project>")
    @app.route("/api/results/meta_data/get/<path:project>")
    @app.route("/api/results/meta_data/get")
    @app.route("/api/results/meta_data/get/")
    def get_meta_data(plate=None, project=None):

        base_url = "/api/results/meta_data/get"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = get_project_name(path)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate in state.enumerate_plates]

            return jsonify(**json_response(
                ["urls"],
                dict(read_only=not lock_key, lock_key=lock_key, is_project=True, project_name=name, urls=urls)))

        if state.meta_data:
            return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                           is_project=True, is_endpoint=True, project_name=name,
                           meta_data=state.meta_data[plate].data)
        else:
            return jsonify(success=False, reason="Project has no meta-data added",
                           read_only=not lock_key, lock_key=lock_key,
                           is_project=True, is_endpoint=True, project_name=name)

    @app.route("/api/results/pinning", defaults={'project': ""})
    @app.route("/api/results/pinning/", defaults={'project': ""})
    @app.route("/api/results/pinning/<path:project>")
    def get_pinning(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/pinning"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        pinnings = list(state.plate_shapes)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        return jsonify(success=True, is_project=True, is_endpoint=True, project_name=get_project_name(path),
                       pinnings=pinnings, plates=sum(1 for p in pinnings if p is not None),
                       read_only=not lock_key, lock_key=lock_key)

    @app.route("/api/results/gridding", defaults={'project': ""})
    @app.route("/api/results/gridding/", defaults={'project': ""})
    @app.route("/api/results/gridding/<int:plate>", defaults={'project': ""})
    @app.route("/api/results/gridding/<int:plate>/<path:project>")
    @app.route("/api/results/gridding/<path:project>")
    def get_gridding(project=None, plate=None):

        base_url = "/api/results/gridding"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, base_url))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = get_project_name(path)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(base_url, plate, project)
                    for plate, shape in enumerate(state.enumerate_plates) if shape is not None]

            return jsonify(**json_response(
                ["urls"],
                dict(
                    read_only=not lock_key, lock_key=lock_key,
                    is_project=True, urls=urls, project_name=name)))

        return send_from_directory(path, Paths().experiment_grid_image_pattern.format(plate + 1))

    @app.route("/api/results/phenotype_names")
    @app.route("/api/results/phenotype_names/")
    @app.route("/api/results/phenotype_names/<path:project>")
    def get_phenotype_names(project=None):

        path = convert_url_to_path(project)
        base_url = "/api/results/phenotype_names"

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(jsonify(is_project=False, **get_search_results(path, base_url)))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
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
                 read_only=not lock_key, lock_key=lock_key, project_name=name)))

    @app.route("/api/results/quality_index")
    @app.route("/api/results/quality_index/")
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
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = get_project_name(path)

        if plate is None:

            urls = ["/api/results/quality_index/{0}/{1}".format(plate, project)
                    for plate, shape in enumerate(state.enumerate_plates) if shape is not None]
            return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                           is_project=True, is_endpoint=False,
                           urls=urls,
                           project_name=name)

        rows, cols = state.get_quality_index(plate)
        return jsonify(success=True, read_only=not lock_key, lock_key=lock_key, is_project=True, is_endpoint=True,
                       project_name=name, dim1_rows=rows.tolist(), dim2_cols=cols.tolist())

    @app.route("/api/results/phenotype")
    @app.route("/api/results/phenotype/")
    @app.route("/api/results/phenotype/<phenotype>/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<phenotype>/<path:project>")
    def get_phenotype_data(phenotype=None, project=None, plate=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(
                ["urls"], dict(is_project=False, **get_search_results(path, "/api/results/phenotype/_NONE_"))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = get_project_name(path)

        if phenotype is None:

            phenotypes = state.phenotype_names()

            if plate is None:

                urls = ["/api/results/phenotype/{0}/{1}/{2}".format(phenotype, plate, project)
                        for phenotype, plate in product(phenotypes, state.enumerate_plates)]
            else:

                urls = ["/api/results/phenotype/{0}/{1}/{2}".format(phenotype, plate, project)
                        for phenotype in phenotypes]

            return jsonify(**json_response(
                ["urls"],
                dict(read_only=not lock_key, lock_key=lock_key, is_project=True, phenotypes=phenotypes, urls=urls,
                     project_name=name)))

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
                    urls=urls, read_only=not lock_key, lock_key=lock_key, project_name=name,
                    plate_indices=plate_indices, is_project=True, is_segmentation_based=is_segmentation_based)))

        plate_data = state.get_phenotype(phenotype_enum)[plate]

        return jsonify(success=True, read_only=not lock_key, lock_key=lock_key, project_name=name,
                       data=plate_data.tojson(), plate=plate, phenotype=phenotype, is_project=True, is_endpoint=True,
                       is_segmentation_based=is_segmentation_based,
                       **{filt.name: tuple(v.tolist() for v in plate_data.where_mask_layer(filt))
                          for filt in Filter if filt != Filter.OK})

    @app.route("/api/results/curves")
    @app.route("/api/results/curves/")
    @app.route("/api/results/curves/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>")
    @app.route("/api/results/curves/<int:plate>/<path:project>")
    @app.route("/api/results/curves/<path:project>")
    def get_growth_data(plate=None, d1_row=None, d2_col=None, project=None):

        url_root = "/api/results/curves"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(**json_response(["urls"], dict(is_project=False, **get_search_results(path, url_root))))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = get_project_name(path)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(**json_response(
                ["urls"],
                dict(read_only=not lock_key, lock_key=lock_key, is_project=True, urls=urls, project_name=name)))

        if d1_row is None or d2_col is None:

            shape = tuple(state.plate_shapes)[plate]
            if shape is None:
                return jsonify(success=False, reason="Plate not included in project")

            urls = ["{0}/{1}/{2}/{3}/{4}".format(url_root, plate, d1, d2, project) for d1, d2 in
                    product(range(shape[0]) if d1_row is None else [d1_row],
                            range(shape[1]) if d2_col is None else [d2_col])]

            return jsonify(**json_response(
                ["urls"],
                dict(read_only=not lock_key, lock_key=lock_key, is_project=True, urls=urls, project_name=name)))

        segmentations = state.get_curve_segments(plate, d1_row, d2_col)
        if segmentations is not None:
            segmentations = segmentations.tolist()

        return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                       is_project=True, is_endpoint=True,
                       project_name=name,
                       time_data=state.times.tolist(),
                       smooth_data=state.smooth_growth_data[plate][d1_row, d2_col].tolist(),
                       raw_data=state.raw_growth_data[plate][d1_row, d2_col].tolist(),
                       segmentations=segmentations)

    # End of UI extension with qc-functionality
    return True
