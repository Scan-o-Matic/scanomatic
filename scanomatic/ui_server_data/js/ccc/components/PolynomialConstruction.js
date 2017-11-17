import PropTypes from 'prop-types';
import React from 'react';

import PolynomialResultsInfo from './PolynomialResultsInfo';


export default function PolynomialConstruction(props) {
    let results = null;
    if (props.error || props.polynomial) {
        results = (
            <PolynomialResultsInfo
                polynomial={props.polynomial}
                error={props.error}
                onClearError={props.onClearError}
            />
        );
    }
    return (
        <div>
            <button
                className="btn btn-default"
                onClick={props.onConstruction}
            >Construct Cell Count Calibration Polynomial</button>
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
