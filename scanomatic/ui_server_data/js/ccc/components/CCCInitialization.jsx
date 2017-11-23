import PropTypes from 'prop-types';
import React from 'react';

export default function CCCInitialization(props) {
    return (
        <div>
            <input
                className="species"
                value={props.species}
                onChange={props.onSpeciesChange}
            />
            <input
                className="reference"
                value={props.reference}
                onChange={props.onReferenceChange}
            />
            <select
                className="fixtures"
                onChange={props.onFixtureNameChange}
                value={props.fixtureName}
            >
                {props.fixtureNames.map(v => (
                    <option key={v} value={v}>{v}</option>
                ))}
            </select>
            <select
                className="pinningformats"
                onChange={props.onPinningFormatNameChange}
                value={props.pinningFormatName}
            >
                {props.pinningFormatNames.map(v => (
                    <option key={v} value={v}>{v}</option>
                ))}
            </select>
            <button onClick={props.onSubmit}>Initiate new CCC</button>
        </div>
    );
}

CCCInitialization.propTypes = {
    species: PropTypes.string.isRequired,
    reference: PropTypes.string.isRequired,
    fixtureName: PropTypes.string.isRequired,
    fixtureNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    pinningFormatName: PropTypes.string.isRequired,
    pinningFormatNames: PropTypes.arrayOf(PropTypes.string).isRequired,
    onSpeciesChange: PropTypes.func.isRequired,
    onReferenceChange: PropTypes.func.isRequired,
    onFixtureNameChange: PropTypes.func.isRequired,
    onPinningFormatNameChange: PropTypes.func.isRequired,
    onSubmit: PropTypes.func.isRequired,
};

CCCInitialization.defaultProps = {
    error: null,
};
