import PropTypes from 'prop-types';
import React from 'react';

export default function NewJob(props) {
    return (
        <div className="row">
            <div className="col-md-6 col-md-offset-3">
                <div className="panel panel-default">
                    <div className="panel-heading">
                        New scan series
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
                                    className="days form-control"
                                    type="number"
                                    value={props.interval}
                                    placeholder="Interval (minutes)"
                                    onChange={props.onIntervalChange}
                                />
                                <span className="input-group-addon" id="interval-unit">minutes</span>
                            </div>
                        </div>
                        <button className="btn btn-primary" onClick={props.onSubmit}>
                            Add to jobs
                        </button>
                        <button className="btn" onClick={props.onCancel}>
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

NewJob.propTypes = {
    name: PropTypes.string,
    error: PropTypes.string,
    duration: PropTypes.shape({
        days: PropTypes.number.isRequired,
        hours: PropTypes.number.isRequired,
        minutes: PropTypes.number.isRequired,
    }).isRequired,
    interval: PropTypes.number.isRequired,
    onNameChange: PropTypes.func.isRequired,
    onDurationDaysChange: PropTypes.func.isRequired,
    onDurationHoursChange: PropTypes.func.isRequired,
    onDurationMinutesChange: PropTypes.func.isRequired,
    onIntervalChange: PropTypes.func.isRequired,
    onSubmit: PropTypes.func.isRequired,
    onCancel: PropTypes.func.isRequired,
};

NewJob.defaultProps = {
    name: '',
    error: null,
};
