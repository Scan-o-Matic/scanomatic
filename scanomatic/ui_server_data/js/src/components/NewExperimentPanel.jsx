import PropTypes from 'prop-types';
import React from 'react';
import myTypes from '../prop-types';

export default function NewExperimentPanel(props) {
    return (
        <div className="row new-experiment-panel">
            <div className="col-md-6">
                <div className="panel panel-default">
                    <div className="panel-heading">
                        New &ldquo;{props.project}&rdquo; experiment
                    </div>
                    {(props.errors && props.errors.general) && (
                        <div className="alert alert-danger" role="alert">
                            {props.errors.general}
                        </div>
                    )}
                    <div className="panel-body">
                        <div className={`form-group${props.errors && props.errors.name ? ' has-error' : ''}`}>
                            <label className="control-label" htmlFor="new-exp-name">Name</label>
                            <input
                                className="form-control"
                                value={props.name}
                                placeholder="Short description of content"
                                onChange={e => props.onNameChange(e.target.value)}
                                name="new-exp-name"
                                data-error={props.errors && props.errors.name}
                            />
                            {(props.errors && props.errors.name) && (
                                <span className="help-block">
                                    {props.errors.name}
                                </span>
                            )}
                        </div>
                        <div className="form-group">
                            <label className="control-label" htmlFor="new-exp-desc">Description</label>
                            <textarea
                                className="description form-control vertical-textarea"
                                placeholder="Full description of experiment and its plates"
                                onChange={e => props.onDescriptionChange(e.target.value)}
                                name="new-exp-desc"
                            >
                                {props.description}
                            </textarea>
                        </div>
                        <div
                            className={`form-group${
                                props.errors && (props.errors.durationDays || props.errors.durationHours || props.errors.durationMinutes) ? ' has-error' : ''
                            }`}
                        >
                            <label className="control-label">Duration</label>
                            <div className="input-group">
                                <input
                                    className="days form-control"
                                    type="number"
                                    value={props.duration.days}
                                    placeholder="Days"
                                    onChange={e => props.onDurationDaysChange(e.target.value)}
                                />
                                <span className="input-group-addon" id="duration-days-unit">days</span>
                            </div>
                            {(props.errors && props.errors.durationDays) && (
                                <span className="help-block">
                                    {props.errors.durationDays}
                                </span>
                            )}
                            <div className="input-group">
                                <input
                                    className="hours form-control"
                                    type="number"
                                    value={props.duration.hours}
                                    placeholder="Hours"
                                    onChange={e => props.onDurationHoursChange(e.target.value)}
                                />
                                <span className="input-group-addon" id="duration-hours-unit">hours</span>
                            </div>
                            {(props.errors && props.errors.durationHours) && (
                                <span className="help-block">
                                    {props.errors.durationHours}
                                </span>
                            )}
                            <div className="input-group">
                                <input
                                    className="minutes form-control"
                                    type="number"
                                    value={props.duration.minutes}
                                    placeholder="Minutes"
                                    onChange={e => props.onDurationMinutesChange(e.target.value)}
                                />
                                <span className="input-group-addon" id="duration-minutes-unit">minutes</span>
                            </div>
                            {(props.errors && props.errors.durationMinutes) && (
                                <span className="help-block">
                                    {props.errors.durationMinutes}
                                </span>
                            )}
                        </div>
                        <div className={`form-group${props.errors && props.errors.interval ? ' has-error' : ''}`}>
                            <label htmlFor="new-exp-interval" className="control-label">Interval</label>
                            <div className="input-group">
                                <input
                                    className="interval form-control"
                                    type="number"
                                    value={props.interval}
                                    placeholder="Interval (minutes)"
                                    onChange={e => props.onIntervalChange(e.target.value)}
                                    name="new-exp-interval"
                                />
                                <span className="input-group-addon" id="interval-unit">minutes</span>
                            </div>
                            {(props.errors && props.errors.interval) && (
                                <span className="help-block">
                                    {props.errors.interval}
                                </span>
                            )}
                        </div>
                        <div className={`form-group${props.errors && props.errors.scanner ? ' has-error' : ''}`}>
                            <label htmlFor="new-exp-scanner" className="control-label">Scanner</label>
                            <select
                                className="scanner form-control"
                                onChange={e => props.onScannerChange(e.target.value)}
                                value={props.scannerId}
                                name="new-exp-scanner"
                            >
                                {props.scanners
                                    .sort((a, b) => {
                                        if (a.name < b.name) return -1;
                                        if (a.name > b.name) return 1;
                                        return 0;
                                    })
                                    .map(v => (
                                        <option key={v.name} value={v.identifier}>
                                            {v.name}
                                            {` (${v.power ? 'online' : 'offline'}, ${v.owned ? 'occupied' : 'free'})`}
                                        </option>
                                    ))}
                            </select>
                            {(props.errors && props.errors.scanner) && (
                                <span className="help-block">
                                    {props.errors.scanner}
                                </span>
                            )}
                        </div>
                        <div className="btn-toolbar" role="toolbar">
                            <button className="btn btn-primary job-add" onClick={props.onSubmit}>
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
    project: PropTypes.string.isRequired,
    errors: PropTypes.shape(myTypes.newExperimentErrorsShape),
    scannerId: PropTypes.string,
    scanners: PropTypes.arrayOf(PropTypes.shape(myTypes.scannerShape)),
    onNameChange: PropTypes.func.isRequired,
    onDescriptionChange: PropTypes.func.isRequired,
    onDurationDaysChange: PropTypes.func.isRequired,
    onDurationHoursChange: PropTypes.func.isRequired,
    onDurationMinutesChange: PropTypes.func.isRequired,
    onIntervalChange: PropTypes.func.isRequired,
    onCancel: PropTypes.func.isRequired,
    onSubmit: PropTypes.func.isRequired,
    ...myTypes.experimentShape,
};

NewExperimentPanel.defaultProps = {
    errors: null,
    scannerId: null,
    scanners: [],
};
