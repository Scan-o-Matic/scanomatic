import PropTypes from 'prop-types';
import React from 'react';
import myTypes from '../prop-types';
import DurationInput from './DurationInput';

const milliSecondsPerMinute = 60000;

export default function NewExperimentPanel(props) {
    return (
        <div className="row new-experiment-panel">
            <div className="col-md-6">
                <div className="panel panel-default">
                    <div className="panel-heading">
                        New &ldquo;{props.projectName}&rdquo; experiment
                    </div>
                    <div className="panel-body">
                        {(props.errors.has('general')) && (
                            <div className="alert general-alert alert-danger" role="alert">
                                {props.errors.get('general')}
                            </div>
                        )}
                        <div className={`form-group group-name ${props.errors.has('name') ? 'has-error' : ''}`}>
                            <label className="control-label" htmlFor="new-exp-name">Name</label>
                            <input
                                className="name form-control"
                                value={props.name}
                                placeholder="Short description of content"
                                onChange={e => props.onChange('name', e.target.value)}
                                name="new-exp-name"
                                data-error={props.errors.get('name')}
                            />
                            {(props.errors.has('name')) && (
                                <span className="help-block">
                                    {props.errors.get('name')}
                                </span>
                            )}
                        </div>
                        <div className={`form-group group-description ${props.errors.has('description') ? 'has-error' : ''}`}>
                            <label className="control-label" htmlFor="new-exp-desc">Description</label>
                            <textarea
                                className="description form-control vertical-textarea"
                                placeholder="Full description of experiment and its plates"
                                onChange={e => props.onChange('description', e.target.value)}
                                name="new-exp-desc"
                                value={props.description}
                            />
                            {(props.errors.has('description')) && (
                                <span className="help-block">
                                    {props.errors.get('description')}
                                </span>
                            )}
                        </div>
                        <DurationInput
                            duration={props.duration}
                            onChange={v => props.onChange('duration', v)}
                            error={props.errors.get('duration')}
                        />
                        <div className={`form-group group-interval ${props.errors.has('interval') ? 'has-error' : ''}`}>
                            <label htmlFor="new-exp-interval" className="control-label">Interval</label>
                            <div className="input-group">
                                <input
                                    className="interval form-control"
                                    type="number"
                                    value={Math.round(props.interval / milliSecondsPerMinute)}
                                    placeholder="Interval (minutes)"
                                    onChange={e => props.onChange('interval', e.target.value * milliSecondsPerMinute)}
                                    name="new-exp-interval"
                                />
                                <span className="input-group-addon" id="interval-unit">minutes</span>
                            </div>
                            {(props.errors.has('interval')) && (
                                <span className="help-block">
                                    {props.errors.get('interval')}
                                </span>
                            )}
                        </div>
                        <div className={`form-group group-scanner ${props.errors.has('scannerId') ? 'has-error' : ''}`}>
                            <label htmlFor="new-exp-scanner" className="control-label">Scanner</label>
                            <select
                                className="scanner form-control"
                                onChange={e => props.onChange('scannerId', e.target.value)}
                                value={props.scannerId}
                                name="new-exp-scanner"
                            >
                                <option key="" value="">
                                    --- Choose scanner ---
                                </option>
                                {props.scanners
                                    .map(v => (
                                        <option key={v.name} value={v.identifier}>
                                            {v.name}
                                            {` (${v.power ? 'online' : 'offline'}, ${v.owned ? 'occupied' : 'free'})`}
                                        </option>
                                    ))}
                            </select>
                            {(props.errors.has('scannerId')) && (
                                <span className="help-block">
                                    {props.errors.get('scannerId')}
                                </span>
                            )}
                        </div>
                        <div className="btn-toolbar" role="toolbar">
                            <button className="btn btn-primary experiment-add" onClick={props.onSubmit}>
                                Add Experiment
                            </button>
                            <button className="btn cancel" onClick={props.onCancel}>
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

NewExperimentPanel.propTypes = {
    projectName: PropTypes.string.isRequired,
    errors: PropTypes.instanceOf(Map),
    scannerId: PropTypes.string,
    scanners: PropTypes.arrayOf(PropTypes.shape(myTypes.scannerShape)),
    onChange: PropTypes.func.isRequired,
    onCancel: PropTypes.func.isRequired,
    onSubmit: PropTypes.func.isRequired,
    ...myTypes.experimentShape,
};

NewExperimentPanel.defaultProps = {
    errors: new Map(),
    scannerId: null,
    scanners: [],
};
