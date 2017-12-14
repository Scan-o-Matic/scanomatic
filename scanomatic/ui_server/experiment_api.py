from __future__ import absolute_import

import os
from types import StringTypes

from flask import request, jsonify
from flask_restful import Api

from scanomatic.io.app_config import Config
from scanomatic.models.compile_project_model import COMPILE_ACTION
from scanomatic.models.factories.analysis_factories import AnalysisModelFactory
from scanomatic.models.factories.compile_project_factory import (
    CompileProjectFactory)
from scanomatic.models.factories.features_factory import FeaturesFactory
from scanomatic.models.factories.scanning_factory import ScanningModelFactory
from scanomatic.util import bioscreen
from scanomatic.data_processing import phenotyper

from .general import get_2d_list, json_abort
from .resources import ScanCollection


def add_routes(app, rpc_client, logger):

    @app.route("/feature_extract", methods=['post'])
    def _feature_extract_api():
        action = request.args.get("action")

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if action:
            if action == 'extract':
                path = data_object.get("analysis_directory")
                path = os.path.abspath(path.replace(
                    'root', Config().paths.projects_root))
                try_keep_qc = bool(
                    data_object.get("keep_qc", default=1, type=int))
                logger.info(
                    "Attempting to extract features in '{0}'".format(path))
                model = FeaturesFactory.create(
                    analysis_directory=path,
                    try_keep_qc=try_keep_qc,
                )

                success = (
                    FeaturesFactory.validate(model) and
                    rpc_client.create_feature_extract_job(
                        FeaturesFactory.to_dict(model)))
                if success:
                    return jsonify(success=True)
                else:
                    return json_abort(
                        400,
                        reason="The following has bad data: {0}".format(
                            ", ".join(
                                FeaturesFactory.get_invalid_names(model)))
                        if not FeaturesFactory.validate(model) else
                        "Refused by the server, check logs.")

            elif action == 'bioscreen_extract':

                path = data_object.get("bioscreen_file")
                path = os.path.abspath(
                    path.replace('root', Config().paths.projects_root))

                if os.path.isfile(path):

                    output = ".".join((path, "features"))

                    try:
                        os.makedirs(output)
                    except OSError:
                        logger.info(
                            "Analysis folder {0} exists, so will overwrite files if needed".format(
                                output))
                else:
                    return json_abort(400, reason="No such file")

                phenotyper.remove_state_from_path(output)
                preprocess = data_object.get(
                    "bioscreen_preprocess", default=None)

                try:
                    preprocess = (
                        bioscreen.Preprocessing(preprocess) if preprocess else
                        bioscreen.Preprocessing.Precog2016_S_cerevisiae)

                except (TypeError, KeyError):
                    return json_abort(
                        400,
                        reason="Unknown pre-processing state")

                time_scale = data_object.get(
                    "bioscreen_timescale", default=36000)
                try:
                    time_scale = float(time_scale)
                except (ValueError, TypeError):
                    return json_abort(400, reason="Bad timescale")

                project = bioscreen.load(
                    path, time_scale=time_scale, preprocess=preprocess)
                project.save_state(output, ask_if_overwrite=False)

                try_keep_qc = bool(
                    data_object.get("try_keep_qc", default=False))

                model = FeaturesFactory.create(
                    analysis_directory=output,
                    extraction_data="State",
                    try_keep_qc=try_keep_qc,
                    )

                success = (
                    FeaturesFactory.validate(model) and
                    rpc_client.create_feature_extract_job(
                        FeaturesFactory.to_dict(model)))

                if success:
                    return jsonify(success=True)
                else:
                    return json_abort(
                        400,
                        reason="The following has bad data: {0}".format(
                            ", ".join(
                                FeaturesFactory.get_invalid_names(model)))
                        if not FeaturesFactory.validate(model) else
                        "Refused by the server, check logs.")
            else:
                return json_abort(
                    400,
                    reason='Action "{0}" not recognized'.format(action))

        return json_abort(
            400,
            reason='Action needed'.format(action))

    @app.route("/analysis", methods=['post'])
    def _analysis_api():

        action = request.args.get("action")

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if action:
            if action == 'analysis':

                path_compilation = data_object.get("compilation")
                path_compilation = os.path.abspath(path_compilation.replace(
                    'root', Config().paths.projects_root))

                path_compile_instructions = data_object.get(
                    "compile_instructions")
                if (path_compile_instructions == "root" or
                        path_compile_instructions == "root/"):
                    path_compile_instructions = None
                elif path_compile_instructions:
                    path_compile_instructions = os.path.abspath(
                        path_compile_instructions.replace(
                            'root', Config().paths.projects_root))

                logger.info(
                    "Attempting to analyse '{0}' (instructions '{1}')".format(
                        path_compilation, path_compile_instructions))

                model = AnalysisModelFactory.create(
                    compilation=path_compilation,
                    compile_instructions=path_compile_instructions,
                    output_directory=data_object.get("output_directory"),
                    cell_count_calibration_id=data_object.get("ccc"),
                    one_time_positioning=bool(data_object.get(
                        'one_time_positioning', default=1, type=int)),
                    chain=bool(data_object.get('chain', default=1, type=int)))

                logger.info(
                    "Created  job model {}".format(
                        AnalysisModelFactory.to_dict(model)))

                if "pinning_matrices" in data_object:
                    model.pinning_matrices = get_2d_list(
                        data_object, "pinning_matrices",
                        getlist_kwargs={"type": int}, dtype=int)

                regridding_folder = data_object.get(
                    "reference_grid_folder", default=None)
                if regridding_folder:
                    grid_list = get_2d_list(
                        data_object, "gridding_offsets",
                        getlist_kwargs={"type": int}, dtype=int)

                    grid_list = tuple(
                        tuple(map(int, l)) if l else None for l in grid_list)

                    model.grid_model.reference_grid_folder = regridding_folder
                    model.grid_model.gridding_offsets = grid_list

                plate_image_inclusion = data_object.getlist(
                    'plate_image_inclusion[]')
                if not plate_image_inclusion:
                    data_object.get('plate_image_inclusion', default=None)

                if plate_image_inclusion:

                    if isinstance(plate_image_inclusion, StringTypes):
                        plate_image_inclusion = tuple(
                            val.strip() for val in
                            plate_image_inclusion.split(";"))
                        plate_image_inclusion = [
                            val if val else None for val in
                            plate_image_inclusion]

                    model.plate_image_inclusion = plate_image_inclusion

                logger.info(
                    "Status `rpc_client.online`{} `rpc_client.local`{}".format(
                        rpc_client.online, rpc_client.local))
                logger.info("Validate model {}".format(
                    AnalysisModelFactory.validate(model)
                ))
                success = (
                    AnalysisModelFactory.validate(model) and
                    rpc_client.create_analysis_job(
                        AnalysisModelFactory.to_dict(model)))

                if success:
                    return jsonify(success=True)
                else:
                    return json_abort(
                        400,
                        reason="The following has bad data: {0}".format(
                            ", ".join(
                                AnalysisModelFactory.get_invalid_names(model))
                            ))

            else:
                return json_abort(
                    400,
                    reason='Action "{0}" not recognized'.format(action))

    @app.route("/experiment", methods=['post'])
    def _experiment_api():

        if request.args.get("enqueue"):

            data_object = request.get_json(silent=True, force=True)
            if not data_object:
                data_object = request.values

            project_name = os.path.basename(
                os.path.abspath(data_object.get("project_path")))
            project_root = os.path.dirname(
                data_object.get("project_path")).replace(
                    'root', Config().paths.projects_root)

            plate_descriptions = data_object.get("plate_descriptions")
            if all(
                    isinstance(p, StringTypes) or p is None
                    for p in plate_descriptions):

                plate_descriptions = tuple(
                    {"index": i, "description": p}
                    for i, p in enumerate(plate_descriptions))

            m = ScanningModelFactory.create(
                number_of_scans=data_object.get("number_of_scans"),
                time_between_scans=data_object.get("time_between_scans"),
                project_name=project_name,
                directory_containing_project=project_root,
                description=data_object.get("description"),
                email=data_object.get("email"),
                pinning_formats=data_object.get("pinning_formats"),
                fixture=data_object.get("fixture"),
                scanner=data_object.get("scanner"),
                scanner_hardware=data_object.get("scanner_hardware")
                if "scanner_hardware" in request.json else "EPSON V700",
                mode=data_object.get("mode", "TPU"),
                plate_descriptions=plate_descriptions,
                cell_count_calibration_id=data_object.get(
                    "cell_count_calibration_id"),
                auxillary_info=data_object.get("auxillary_info"),
            )

            validates = ScanningModelFactory.validate(m)

            job_id = rpc_client.create_scanning_job(
                ScanningModelFactory.to_dict(m))

            if validates and job_id:
                return jsonify(success=True, name=project_name)
            else:

                return jsonify(
                    success=False,
                    reason="The following has bad data: {0}".format(
                        ScanningModelFactory.get_invalid_as_text(m))
                    if not validates else
                    "Job refused, probably scanner can't be reached, check connection.")

    @app.route("/compile", methods=['post'])
    def _compile_api():

        data_object = request.get_json(silent=True, force=True)
        if not data_object:
            data_object = request.values

        if request.args.get("run"):

            if not rpc_client.online:
                return jsonify(
                    success=False,
                    reason="Scan-o-Matic server offline")

            path = request.values.get('path')
            path = os.path.abspath(
                path.replace('root', Config().paths.projects_root))
            fixture_is_local = bool(int(data_object.get('local')))
            fixture = data_object.get("fixture")
            chain_steps = bool(data_object.get('chain', default=1, type=int))
            images = data_object.getlist('images[]')

            logger.info(
                "Attempting to compile on path {0}, as {1} fixture{2} (Chaining: {3}), images {4}".format(
                    path,
                    'local' if fixture_is_local else 'global',
                    fixture_is_local and "." or " (Fixture {0}).".format(
                        fixture),
                    chain_steps, images))

            dict_model = CompileProjectFactory.dict_from_path_and_fixture(
                path, fixture=fixture, is_local=fixture_is_local,
                compile_action=COMPILE_ACTION.InitiateAndSpawnAnalysis
                if chain_steps else COMPILE_ACTION.Initiate)

            n_images_in_folder = len(dict_model['images'])

            if images:

                dict_model['images'] = [
                    p for p in dict_model['images']
                    if os.path.basename(p['path']) in images
                ]

                app.logger.info(
                    "Manual selection of images, {0} included of {1} requested compared to {2} in folder.".format(
                        len(dict_model['images']), len(images),
                        n_images_in_folder))

                if len(dict_model['images']) != len(images):
                    return jsonify(
                        success=False,
                        reason="The manually set list of images could not be satisfied"
                        "with the images in the specified folder")
            else:

                app.logger.info(
                    "Using all {0} images in folder for compilation".format(
                        n_images_in_folder))

            dict_model["overwrite_pinning_matrices"] = get_2d_list(
                data_object, "pinning_matrices",
                getlist_kwargs={"type": int}, dtype=int)

            job_id = rpc_client.create_compile_project_job(dict_model)

            return jsonify(
                success=True if job_id else False,
                reason="" if job_id else "Invalid parameters")

    api = Api(app)
    api.add_resource(ScanCollection, '/api/scans', endpoint='scans')
