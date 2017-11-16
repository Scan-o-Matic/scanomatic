import PropTypes from 'prop-types';
import React from 'react';

import PolyConstuctionButton from './PolyConstructionButton';
import PolyResults from './PolyResults';


export default function Polynomial(props) {
    return (
        <div>
            <PolyConstuctionButton
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

Polynomial.propTypes = {
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
