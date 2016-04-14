import os
import time
import uuid
from flask import request, Flask, jsonify
from itertools import chain

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
        return _add_lock(path)


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
        is_results = phenotyper.path_has_saved_project_state(path)

        return jsonify(success=True,
                       project=project,
                       is_results=is_results,
                       **_get_search_results(path, "/api/results/browse"))

    @app.route("/api/results/lock/add/<path:project>")
    def lock_project(project=""):

        path = convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")
        key = _validate_lock_key(path, "")
        if key:
            return jsonify(success=True, lock_key=key)
        else:
            return jsonify(success=False, reason="Someone else is working with these results")

    @app.route("/api/results/lock/remove/<path:project>")
    def unlock_project(project=""):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")

        if not _validate_lock_key(path, request.form['lock_key']):
            return jsonify(success=False, reason="Invalid key")

        _remove_lock(path)
        return jsonify(success=True)

    @app.route("/api/results/meta_data/add/<path:project>", methods=["POST"])
    def add_meta_data(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=True,
                           is_results=False,
                           **_get_search_results(path, "/api/results/meta_data/add/"))

        if not _validate_lock_key(path, request.form["lock_key"]):
            return jsonify(success=False, reason="Someone else is working with these results")

        data = request.form["file"]
        file_sufix = request.form["file_suffix"]

        meta_data_path = _get_new_metadata_file_name(path, file_sufix)
        with open(meta_data_path, 'wb') as fh:
            fh.write(data)
        state = phenotyper.Phenotyper.LoadFromState(path)
        state.load_meta_data(meta_data_path)
        state.save_state(path, ask_if_overwrite=False)

        return jsonify(success=True)

    @app.route("/api/results/pinning", defaults={'project': ""})
    @app.route("/api/results/pinning/", defaults={'project': ""})
    @app.route("/api/results/pinning/<path:project>")
    def get_pinning(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_results=False,
                           **_get_search_results(path, "/api/results/pinning"))

        state = phenotyper.Phenotyper.LoadFromState(path)
        pinnings = list(state.plate_shapes)
        read_only = _validate_lock_key(path, request.form["lock_key"])
        return jsonify(success=True, is_results=True, project_name=_get_project_name(path),
                       pinnings=pinnings, plates=sum(1 for p in pinnings if p is not None),
                       read_only=read_only)

    @app.route("/api/results/phenotype_names")
    @app.route("/api/results/phenotype_names/")
    @app.route("/api/results/phenotype_names/<path:project>")
    def get_phenotype_names(project=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_results=False,
                           **_get_search_results(path, "/api/results/phenotype_names"))

        state = phenotyper.Phenotyper.LoadFromState(path)
        read_only = _validate_lock_key(path, request.form["lock_key"])
        name = _get_project_name(path)

        return jsonify(success=True, phenotypes=state.phenotype_names(),
                       read_only=read_only, project_name=name)

    @app.route("/api/results/phenotype")
    @app.route("/api/results/phenotype/")
    @app.route("/api/results/phenotype/<phenotype>/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<phenotype>/<path:project>")
    @app.route("/api/results/phenotype/<int:plate>/<path:project>")
    @app.route("/api/results/phenotype/<path:project>")
    def get_phenotype_data(phenotype=None, project=None, plate=None):

        path = convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):

            return jsonify(success=True,
                           is_results=False,
                           **_get_search_results(path, "/api/results/phenotype"))

        state = phenotyper.Phenotyper.LoadFromState(path)
        read_only = _validate_lock_key(path, request.form["lock_key"])
        name = _get_project_name(path)

        if phenotype is None:

            phenotypes = state.phenotype_names()

            # TODO: Add some smart urls about phenotypes including platees if exists
            return jsonify(success=True, read_only=read_only, phenotypes=phenotypes,
                           project_name=name)

        if plate is None:

            urls = ["/api/results/phenotype/{0}/{1}".format(i + 1, project)
                    for i, p in enumerate(state.plate_shapes) if p is not None]

            return jsonify(success=True, urls=urls, read_only=read_only, project_name=name)

        phenotype_enum = phenotyper.get_phenotype(phenotype)
        data = state.get_phenotype(phenotype_enum)[plate].filled()

        return jsonify(success=True, read_only=read_only, project_name=name, data=data, plate=plate,
                       phenotype=phenotype)

    @app.route("/api/results/phenotypes/<int:plate>/<int:pos_x>/<int:pos_y>/<path:project>")
    def get_phenotypes_for_position():

        return jsonify(success=False, reason="Not implemented")

    return True
