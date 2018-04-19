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
                    {props.error && (
                        <div className="alert alert-danger" role="alert">
                            {props.error}
                        </div>
                    )}
                    <div className="panel-body">
                        <div className="form-group">
                            <label>Name</label>
                            <input
                                className="name form-control"
                                value={props.name}
                                placeholder="Short description of content"
                                onChange={props.onNameChange}
                            />
                        </div>
                        <div className="form-group">
                            <label>Description</label>
                            <textarea
                                className="name form-control vertical-textarea"
                                placeholder="Full description of experiment and its plates"
                                onChange={props.onDescriptionChange}
                            >
                                {props.description}
                            </textarea>
                        </div>
                        <div className="form-group">
                            <label>Duration</label>
                            <div className="input-group">
                                <input
                                    className="days form-control"
                                    type="number"
                                    value={props.duration.days}
                                    placeholder="Days"
                                    onChange={props.onDurationDaysChange}
                                />
                                <span className="input-group-addon" id="duration-days-unit">days</span>
                            </div>
                            <div className="input-group">
                                <input
                                    className="hours form-control"
                                    type="number"
                                    value={props.duration.hours}
                                    placeholder="Hours"
                                    onChange={props.onDurationHoursChange}
                                />
                                <span className="input-group-addon" id="duration-hours-unit">hours</span>
                            </div>
                            <div className="input-group">
                                <input
                                    className="minutes form-control"
                                    type="number"
                                    value={props.duration.minutes}
                                    placeholder="Minutes"
                                    onChange={props.onDurationMinutesChange}
                                />
                                <span className="input-group-addon" id="duration-minutes-unit">minutes</span>
                            </div>
                        </div>
                        <div className="form-group">
                            <label>Interval</label>
                            <div className="input-group">
                                <input
                                    className="interval form-control"
                                    type="number"
                                    value={props.interval}
                                    placeholder="Interval (minutes)"
                                    onChange={props.onIntervalChange}
                                />
                                <span className="input-group-addon" id="interval-unit">minutes</span>
                            </div>
                        </div>
                        <button className="btn btn-primary job-add" onClick={props.onSubmit}>
                            Add to jobs
                        </button>
                        <button className="btn cancel" onClick={props.onCancel}>
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

NewExperimentPanel.propTypes = {
    project: PropTypes.string.isRequired,
    error: PropTypes.string,
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
    error: null,
};
