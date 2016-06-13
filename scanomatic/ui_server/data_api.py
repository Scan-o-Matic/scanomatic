from flask import request, Flask, jsonify
from types import ListType
import numpy as np

from scanomatic.data_processing import phenotyper
from scanomatic.image_analysis.grayscale import getGrayscales, getGrayscale


def _depth(arr, lvl=1):

    if isinstance(arr, ListType) and len(arr) and isinstance(arr[0], ListType):
        _depth(arr[0], lvl + 1)
    else:
        return lvl


def _validate_depth(data):

    depth = _depth(data)

    while depth < 4:
        data = [data]
        depth += 1

    return data


def json_data(data):

    if data is None:
        return None
    if isinstance(data, ListType):
        return [json_data(d) for d in data]
    elif hasattr(data, "tolist"):
        return data.tolist()
    else:
        return data


def add_routes(app):

    """

    Args:
        app (Flask): The flask app to decorate
    """

    @app.route("/api/data/phenotype", methods=['POST'])
    @app.route("/api/data/phenotype/", methods=['POST'])
    def data_phenotype():
        """Takes growth data extracts phenotypes and normalizes.

        _NOTE_: This API only accepts json-formatted POST calls.

        Minimal example of request json:
        ```
            {"raw_growth_data": [1.23, 1.52 ... 4.24],
             "times_data": [0, 0.33, 0.67 ... 72]}
        ```

        Mandatory keys:

            "raw_growth_data", a 4-dimensional array where the outer dimension
                represents plates, the two middle represents a plate layout and
                the fourth represents the data vector/time dimension.
                It is possible to send a single curve in a 1-d array or arrays with
                2 or 3 dimensions. These will be reshaped with new dimensions added
                outside of the existing dimensions until i has 4 dimensions.

            "times_data", a time vector with the same length as the inner-most
                dimension of "raw_growth_data". Time should be given in hours.

        Optional keys:

            "smooth_growth_data" must match in shape with "raw_growth_data" and
                will trigger skipping the default curve smoothing.
            "inclusion_level" (Trusted, UnderDevelopment, Other) default is
                Trusted.
            "normalize", (True, False), default is False
            "settings" a key: value type array/dictionary which can contain
                the following keys:

                    "median_kernel_size", odd value, default 5
                    "gaussian_filter_sigma", float, default 1.5
                    "linear_regression_size", odd value, default 5

            "reference_offset" str, is optional and only active if "normalize" is
                set to true. Any of the following (UpperLeft, UpperRight,
                LowerLeft, LowerRight).

        :return: json-object with analyzed data
        """

        raw_growth_data = request.json.get("raw_growth_data", [])
        times_data = request.json.get("times_data", [])
        settings = request.json.get("settings", {})
        smooth_growth_data = request.json.get("smooth_growth_data", [])
        inclusion_level = request.json.get("inclusion_level", "Trusted")
        normalize = request.json.get("normalize", False)
        reference_offset = request.json.get("reference_offset", "LowerRight")
        raw_growth_data = _validate_depth(raw_growth_data)

        state = phenotyper.Phenotyper(
            np.array(raw_growth_data), np.array(times_data), run_extraction=False, **settings)

        if smooth_growth_data:
            smooth_growth_data = _validate_depth(smooth_growth_data)
            state.set("smooth_growth_data", np.array(smooth_growth_data))

        state.set_phenotype_inclusion_level(phenotyper.PhenotypeDataType[inclusion_level])
        state.extract_phenotypes(resmoothen=len(smooth_growth_data) == 0)

        curve_segments = state.curve_segments

        if normalize:
            state.set_control_surface_offsets(phenotyper.Offsets[reference_offset])

            return jsonify(smooth_growth_data=json_data(state.smooth_growth_data),
                           phenotypes={
                               pheno.name: [None if p is None else p.tojson() for p in state.get_phenotype(pheno)]
                               for pheno in state.phenotypes},
                           phenotypes_normed={
                               pheno.name: [p.tojson() for p in state.get_phenotype(pheno, normalized=True)]
                               for pheno in state.phenotypes_that_normalize},
                           curve_phases=json_data(curve_segments))

        return jsonify(smooth_growth_data=json_data(state.smooth_growth_data),
                       phenotypes={
                           pheno.name: [None if p is None else p.tojson() for p in state.get_phenotype(pheno)]
                           for pheno in state.phenotypes},
                       curve_phases=json_data(curve_segments))

    @app.route("/api/data/grayscales", methods=['post', 'get'])
    @app.route("/api/data/grayscales/", methods=['post', 'get'])
    def _grayscales():

        return jsonify(grayscales=getGrayscales())

    # End of adding routes
