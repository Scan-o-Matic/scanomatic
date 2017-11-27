import PropTypes from 'prop-types';
import React from 'react';

import PolynomialResultsInfo from './PolynomialResultsInfo';
import PolynomialConstructionError from './PolynomialConstructionError';
import PolynomialResultsPlotScatter from './PolynomialResultsPlotScatter';


export default function PolynomialConstruction(props) {
    let error = null;
    if (props.error) {
        error = (
            <PolynomialConstructionError
                error={props.error}
                onClearError={props.onClearError}
            />
        );
    }

    let resultsInfo = null;
    if (props.polynomial) {
        resultsInfo = (
            <PolynomialResultsInfo
                polynomial={props.polynomial}
            />
        );
    }

    let resultsScatter = null;
    if (props.resultsData) {
        resultsScatter = (
            <PolynomialResultsPlotScatter
                resultsData={props.resultsData}
                correlation={props.correlation}
            />
        );
    }

    return (
        <div>
            <button
                className="btn btn-default btn-construct"
                onClick={props.onConstruction}
            >Construct Cell Count Calibration Polynomial</button>
            <button
                className="btn btn-success btn-finalize"
                disabled={!props.polynomial}
                onClick={props.onFinalizeCCC}
            >
                Finalize and publish calibration
            </button>
            {error}
            {resultsInfo}
            {resultsScatter}
        </div>
    );
}

PolynomialConstruction.propTypes = {
    onConstruction: PropTypes.func.isRequired,
    onClearError: PropTypes.func.isRequired,
    onFinalizeCCC: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
        power: PropTypes.number.isRequired,
        coefficients: PropTypes.array.isRequired,
        colonies: PropTypes.number.isRequired,
    }),
    resultsData: PropTypes.shape({
        calculated: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
    }),
    correlation: PropTypes.shape({
        slope: PropTypes.number.isRequired,
        intercept: PropTypes.number.isRequired,
        stderr: PropTypes.number.isRequired,
    }),
    error: PropTypes.string,
};
