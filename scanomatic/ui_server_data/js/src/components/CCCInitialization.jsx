import PropTypes from 'prop-types';
import React from 'react';

export default function CCCInitialization(props) {
    return (
        <div className="row">
            <div className="col-md-6 col-md-offset-3">
                <div className="panel panel-default">
                    <div className="panel-heading">
                        Initiate New CCC
                    </div>
                    <div className="panel-body">
                        <div className="form-group">
                            <label>Species</label>
                            <input
                                className="species form-control"
                                value={props.species}
                                placeholder="species"
                                onChange={props.onSpeciesChange}
                            />
                        </div>
                        <div className="form-group">
                            <label>Reference</label>
                            <input
                                className="reference form-control"
                                placeholder="reference"
                                value={props.reference}
                                onChange={props.onReferenceChange}
                            />
                        </div>
                        <div className="form-group">
                            <label>Fixture</label>
                            <select
                                className="fixtures form-control"
                                onChange={props.onFixtureNameChange}
                                value={props.fixtureName}
                            >
                                {props.fixtureNames.map(v => (
                                    <option key={v} value={v}>{v}</option>
                                ))}
                            </select>
                        </div>
                        <div className="form-group">
                            <label>Pinning Format</label>
                            <select
                                className="pinningformats form-control"
                                onChange={props.onPinningFormatNameChange}
                                value={props.pinningFormatName}
                            >
                                {props.pinningFormatNames.map(v => (
                                    <option key={v} value={v}>{v}</option>
                                ))}
                            </select>
                        </div>
                        <button className="btn btn-primary" onClick={props.onSubmit}>
                            Initiate new CCC
                        </button>
                    </div>
                </div>
            </div>
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
