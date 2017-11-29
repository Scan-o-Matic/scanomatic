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

    const degrees = ['2', '3', '4', '5'];
    return (
        <div>
            <h3>Cell Count Calibration Polynomial</h3>
            <div className="form-inline">
                <div className="form-group">
                    <label>Degree of polynomial</label>
                    <select
                        className="degree form-control"
                        onChange={props.onDegreeOfPolynomialChange}
                        value={props.degreeOfPolynomial}
                    >
                        {degrees.map(d => <option key={d} value={d}>{d}</option>)}
                    </select>
                </div>
                {' '}
                <button
                    className="btn btn-default btn-construct"
                    onClick={props.onConstruction}
                >
                    Construct Polynomial
                </button>
            </div>
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
    degreeOfPolynomial: PropTypes.number.isRequired,
    onConstruction: PropTypes.func.isRequired,
    onClearError: PropTypes.func.isRequired,
    onDegreeOfPolynomialChange: PropTypes.func.isRequired,
    onFinalizeCCC: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
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

PolynomialConstruction.defaultProps = {
    polynomial: null,
    resultsData: null,
    correlation: null,
    error: null,
};
