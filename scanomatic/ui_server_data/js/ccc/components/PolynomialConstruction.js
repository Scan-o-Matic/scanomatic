import PropTypes from 'prop-types';
import React from 'react';

import PolyConstructionButton from './PolyConstructionButton';
import PolyResults from './PolyResults';


export default function PolynomialConstruction(props) {
    return (
        <div>
            <PolyConstructionButton
                onConstruction={props.onConstruction}
                power={props.power}
            />
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
    power: PropTypes.number.isRequired,
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
