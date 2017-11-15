import PropTypes from 'prop-types';
import React from 'react';

export default function PolyConstuctionButton(props) {
    return (
        <button
            className="btn btn-default"
            onClick={props.handleConstruction}
        >Construct {props.power}th Degree Calibration Polynomial</button>
    );
}

PolyConstuctionButton.propTypes = {
    handleConstruction: PropTypes.func.isRequired,
    power: PropTypes.number.isRequired,
};
