import PropTypes from 'prop-types';
import React from 'react';

import PolyResults from './PolyResults';


export default function PolynomialConstruction(props) {
    return (
        <div>
            <button
                className="btn btn-default"
                onClick={props.onConstruction}
            >Construct Cell Count Calibration Polynomial</button>
            <PolyResults
                polynomial={props.polynomial}
                data={props.data}
                error={props.error}
                onClearError={props.onClearError}
            />
        </div>
    );
}

PolynomialConstruction.propTypes = {
    onConstruction: PropTypes.func.isRequired,
    onClearError: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
        power: PropTypes.number.isRequired,
        coefficients: PropTypes.array.isRequired,
    }),
    data: PropTypes.shape({
        calculated: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
    }),
    error: PropTypes.string,
};
