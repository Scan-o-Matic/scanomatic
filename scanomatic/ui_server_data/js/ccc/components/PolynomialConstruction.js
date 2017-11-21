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

    return (
        <div>
            <button
                className="btn btn-default"
                onClick={props.onConstruction}
            >Construct Cell Count Calibration Polynomial</button>
            {error}
            {results}
        </div>
    );
}

PolynomialConstruction.propTypes = {
    onConstruction: PropTypes.func.isRequired,
    onClearError: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
        power: PropTypes.number.isRequired,
        coefficients: PropTypes.array.isRequired,
        colonies: PropTypes.number.isRequired,
    }),
    error: PropTypes.string,
};
