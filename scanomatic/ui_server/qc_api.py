from flask import request, Flask, jsonify
from scanomatic.dataProcessing import phenotyper
from scanomatic.io.paths import Paths
import os
import uuid


def _add_lock(path):

    key = uuid.uuid4().hex
    _update_lock(path, key)
    return key


def _update_lock(path, key):

    pass


def _remove_lock(path):

    return True


def _validate_lock_key(path, key=""):
    # 1 Translate project to real path
    # 2 If non-expired lock validate same
    if not True:
        return ""

    if key:
        return key
    else:
        return _add_lock(path)


def _discover_projects(path):

    return []


def _get_project_name(projects):

    return []


def _convert_path_to_url(prefix, path):
    # TODO: Strip root/jail from path
    return "/".join((prefix, path))


def _convert_url_to_path(url):

    return url


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
    urls = list(_convert_path_to_url(url_prefix, p) for p in projects)

    return {'names': names, 'urls': urls}


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/results/browse/<project>")
    def browse_for_results(project=""):

        path = _convert_url_to_path(project)
        is_results = phenotyper.path_has_saved_project_state(path)

        return jsonify(success=True,
                       project=project,
                       is_results=is_results,
                       **_get_search_results(path, "/api/results/browse"))

    @app.route("/api/results/lock/add/<project>")
    def lock_project(project=""):

        path = _convert_url_to_path(project)
        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")
        key = _validate_lock_key(path, "")
        if key:
            return jsonify(success=True, lock_key=key)
        else:
            return jsonify(success=False, reason="Someone else is working with these results")

    @app.route("/api/results/lock/remove/<project>")
    def lock_project(project=""):

        path = _convert_url_to_path(project)

        if not phenotyper.path_has_saved_project_state(path):
            return jsonify(success=False, reason="Not a project")

        if not _validate_lock_key(path, request.form['lock_key']):
            return jsonify(success=False, reason="Invalid key")

        _remove_lock(path)
        return jsonify(success=True)

    @app.route("/api/results/meta_data/add/<project>", methods=["POST"])
    def add_meta_data(project=None):

        path = _convert_url_to_path(project)

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
    @app.route("/api/results/pinning/<project>")
    def get_pinning(project=None):

        path = _convert_url_to_path(project)

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

    @app.route("/api/results/phenotype_names/")
    @app.route("/api/results/phenotype_names/<project>")
    def get_phenotype_names(project=None):

        path = _convert_url_to_path(project)

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
    @app.route("/api/results/phenotype/<phenotype>/<int: plate>/<project>")
    @app.route("/api/results/phenotype/<phenotype>/<project>")
    @app.route("/api/results/phenotype/<int: plate>/<project>")
    @app.route("/api/results/phenotype/<project>")
    def get_phenotype_data(phenotype=None, project=None, plate=None):

        path = _convert_url_to_path(project)

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

    @app.route("/api/results/phenotypes/<int: plate>/<int: pos_x>/<int: pos_y>/<project>")
    def get_phenotypes_for_position():

        return jsonify(success=False, reason="Not implemented")

    return True
