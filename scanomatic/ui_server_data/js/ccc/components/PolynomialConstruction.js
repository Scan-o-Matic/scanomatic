import PropTypes from 'prop-types';
import React from 'react';

import PolynomialResultsInfo from './PolynomialResultsInfo';
import PolynomialConstructionError from './PolynomialConstructionError';


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

    let results = null;
    if (props.polynomial) {
        results = (
            <PolynomialResultsInfo
                polynomial={props.polynomial}
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
                    className="btn btn-default"
                    onClick={props.onConstruction}
                >
                    Construct Polynomial
                </button>
            </div>
            {error}
            {results}
        </div>
    );
}

PolynomialConstruction.propTypes = {
    degreeOfPolynomial: PropTypes.number.isRequired,
    onConstruction: PropTypes.func.isRequired,
    onClearError: PropTypes.func.isRequired,
    onDegreeOfPolynomialChange: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
        coefficients: PropTypes.array.isRequired,
        colonies: PropTypes.number.isRequired,
    }),
    error: PropTypes.string,
};
