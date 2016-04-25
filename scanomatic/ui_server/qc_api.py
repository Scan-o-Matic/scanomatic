import os
import time
import uuid
from flask import request, Flask, jsonify
from itertools import chain, product

from scanomatic.dataProcessing import phenotyper
from scanomatic.io.paths import Paths
from scanomatic.ui_server.general import convert_url_to_path, convert_path_to_url

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


def _get_project_name(project_path):
    # TODO: Implement this
    return "Unknown/Not implemented"


def _get_new_metadata_file_name(project_path, suffix):

    i = 1
    while True:
        path = os.path.join(project_path,
                            Paths().phenotypes_meta_data_original_file_patern.format(i, suffix))
        i += 1
        if not os.path.isfile(path):
            return path


def _get_search_results(path, url_prefix):

    projects = _discover_projects(path)
    names = list(_get_project_name(p) for p in projects)
    urls = list(convert_path_to_url(url_prefix, p) for p in projects)
    if None in urls:
        try:
            names, urls = zip(*tuple((n, u) for n, u in zip(names, urls) if u is not None))
        except ValueError:
            pass
    return {'names': names, 'urls': urls}


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/results/browse")
    @app.route("/api/results/browse/")
    @app.route("/api/results/browse/<path:project>")
    def browse_for_results(project=""):

        path = convert_url_to_path(project)
        print path
        is_project = phenotyper.path_has_saved_project_state(path)

        return jsonify(success=True,
                       project=project,
                       is_project=is_project,
                       is_endpoint=False,
                       **_get_search_results(path, "/api/results/browse"))

    @app.route("/api/results/lock/add/<path:project>")
    def lock_project(project=""):

        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")
        key = _validate_lock_key(path, "")
        if key:
            return jsonify(success=True, is_project=True, is_endpoint=True, lock_key=key)
        else:
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Someone else is working with these results")

    @app.route("/api/results/lock/remove/<path:project>")
    def unlock_project(project=""):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, is_project=False, is_endpoint=True, reason="Not a project")

        if not _validate_lock_key(path, request.form['lock_key']):
            return jsonify(success=False, is_project=True, is_endpoint=True, reason="Invalid key")

        _remove_lock(path)
        return jsonify(success=True)

    @app.route("/api/results/meta_data/add/<path:project>", methods=["POST"])
    def add_meta_data(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=True,
                           is_project=False,
                           is_endpoint=False,
                           **_get_search_results(path, "/api/results/meta_data/add/"))

        if not _validate_lock_key(path, request.form["lock_key"]):
            return jsonify(success=False, is_project=True, is_endpoint=True,
                           reason="Someone else is working with these results")

        data = request.form["file"]
        file_sufix = request.form["file_suffix"]

        meta_data_path = _get_new_metadata_file_name(path, file_sufix)
        with open(meta_data_path, 'wb') as fh:
            fh.write(data)
        state = phenotyper.Phenotyper.LoadFromState(path)
        state.load_meta_data(meta_data_path)
        state.save_state(path, ask_if_overwrite=False)

        return jsonify(success=True, is_project=True, is_endpoint=True)

    @app.route("/api/results/pinning", defaults={'project': ""})
    @app.route("/api/results/pinning/", defaults={'project': ""})
    @app.route("/api/results/pinning/<path:project>")
    def get_pinning(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_project=False,
                           is_endpoint=False,
                           **_get_search_results(path, "/api/results/pinning"))

        state = phenotyper.Phenotyper.LoadFromState(path)
        pinnings = list(state.plate_shapes)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        return jsonify(success=True, is_project=True, is_endpoint=True, project_name=_get_project_name(path),
                       pinnings=pinnings, plates=sum(1 for p in pinnings if p is not None),
                       read_only=not lock_key, lock_key=lock_key)

    @app.route("/api/results/phenotype_names")
    @app.route("/api/results/phenotype_names/")
    @app.route("/api/results/phenotype_names/<path:project>")
    def get_phenotype_names(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_project=False,
                           **_get_search_results(path, "/api/results/phenotype_names"))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = _get_project_name(path)

        return jsonify(success=True, phenotypes=state.phenotype_names(),
                       is_project=True, is_endpoint=True,
                       read_only=not lock_key, lock_key=lock_key, project_name=name)

    @app.route("/api/results/phenotype")
    @app.route("/api/results/phenotype/")
    @app.route("/api/results/phenotype/<phenotype>/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<phenotype>/<path:project>")
    @app.route("/api/results/phenotype/<int:plate>/<path:project>")
    def get_phenotype_data(phenotype=None, project=None, plate=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_project=False,
                           is_endpoint=False,
                           **_get_search_results(path, "/api/results/phenotype"))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = _get_project_name(path)

        if phenotype is None:

            phenotypes = state.phenotype_names()

            # TODO: Add some smart urls about phenotypes including plates if exists
            return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                           is_project=True, is_endpoint=False,
                           phenotypes=phenotypes, urls=[],
                           project_name=name)

        if plate is None:

            urls = ["/api/results/phenotype/{0}/{1}".format(i + 1, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(success=True, urls=urls, read_only=not lock_key, lock_key=lock_key, project_name=name,
                           is_project=True, is_endpoint=False)

        phenotype_enum = phenotyper.get_phenotype(phenotype)
        data = state.get_phenotype(phenotype_enum)[plate].filled()

        return jsonify(success=True, read_only=not lock_key, lock_key=lock_key, project_name=name, data=data.tolist(),
                       plate=plate, phenotype=phenotype, is_project=True, is_endpoint=True)

    @app.route("/api/results/curves")
    @app.route("/api/results/curves/")
    @app.route("/api/results/curves/<int:plate>/<int:d1_row>/<int:d2_col>/<path:project>")
    @app.route("/api/results/curves/<int:plate>/<path:project>")
    @app.route("/api/results/curves/<path:project>")
    def get_growth_data(plate=None, d1_row=None, d2_col=None, project=None):

        url_root = "/api/results/curves"
        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_project=False,
                           is_endpoint=False,
                           **_get_search_results(path, url_root))

        state = phenotyper.Phenotyper.LoadFromState(path)
        lock_key = _validate_lock_key(path, request.values.get("lock_key"))
        name = _get_project_name(path)

        if plate is None:

            urls = ["{0}/{1}/{2}".format(url_root, i + 1, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                           is_project=True, is_endpoint=False, urls=urls,
                           project_name=name)

        if d1_row is None or d2_col is None:

            shape = tuple(state.plate_shapes)[plate - 1]
            if shape is None:
                return jsonify(success=False, reason="Plate not included in project")

            urls = ["{0}/{1}/{2}/{3}/{4}".format(url_root, plate, d1, d2, project) for d1, d2 in
                    product(range(shape[0]) if d1_row is None else [d1_row],
                            range(shape[1]) if d2_col is None else [d2_col])]

            return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                           is_project=True, is_endpoint=False, urls=urls,
                           project_name=name)

        return jsonify(success=True, read_only=not lock_key, lock_key=lock_key,
                       is_project=True, is_endpoint=True,
                       project_name=name,
                       time_data=state.times,
                       smooth_data=state.smooth_growth_data[plate - 1][d1_row, d2_col].tolist(),
                       raw_data=state.raw_growth_data[plate - 1][d1_row, d2_col].tolist())

    # End of UI extension with qc-functionality
    return True
