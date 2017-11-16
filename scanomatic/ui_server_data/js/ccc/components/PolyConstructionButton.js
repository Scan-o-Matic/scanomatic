import PropTypes from 'prop-types';
import React from 'react';

export default function PolyConstructionButton(props) {
    return (
        <button
            className="btn btn-default"
            onClick={props.onConstruction}
        >Construct {props.power}th Degree Calibration Polynomial</button>
    );
}

PolyConstructionButton.propTypes = {
    onConstruction: PropTypes.func.isRequired,
    power: PropTypes.number.isRequired,
};
