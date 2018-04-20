import PropTypes from 'prop-types';
import React from 'react';
import Duration from '../Duration';

export default class DurationInput extends React.Component {
    constructor(props) {
        super(props);
        this.handleDaysChange = this.handleDaysChange.bind(this);
        this.handleHoursChange = this.handleHoursChange.bind(this);
        this.handleMinutesChange = this.handleMinutesChange.bind(this);
        this.handleBlur = this.handleBlur.bind(this);
        const duration = Duration.fromMilliseconds(this.props.duration);
        this.state = {
            duration,
        };
    }
    handleDaysChange(evt) {
        const val = Number.parseInt(evt.target.value, 10);
        if (!Number.isNaN(val)) this.setState({ days: val });
    }

    handleHoursChange(evt) {
        const val = Number.parseInt(evt.target.value, 10);
        if (!Number.isNaN(val)) this.setState({ hours: val });
    }

    handleMinutesChange(evt) {
        const val = Number.parseInt(evt.target.value, 10);
        if (!Number.isNaN(val)) this.setState({ minutes: val });
    }

    handleBlur() {
        const {
            duration, days, hours, minutes,
        } = this.state;
        const newDuration = duration.shifted(days || 0, hours || 0, minutes || 0);
        const value = Math.max(newDuration.totalMilliseconds, 0);
        if (value !== this.props.duration) {
            if (this.props.onChange) this.props.onChange(value);
            this.setState({
                days: newDuration.days,
                hours: newDuration.hours,
                minutes: newDuration.minutes,
            });
        }
    }

    render() {
        const { duration, newDuration } = this.state;
        let { days, hours, minutes } = this.state;
        if (days == null) days = newDuration ? newDuration.days : duration.days;
        if (hours == null) hours = newDuration ? newDuration.hours : duration.hours;
        if (minutes == null) minutes = newDuration ? newDuration.minutes : duration.minutes;
        if (days === 0) days = '';
        if (hours === 0) hours = '';
        if (minutes === 0) minutes = '';
        return (
            <div className={`form-group${this.props.error ? ' has-error' : ''}`}>
                <label className="control-label">Duration</label>
                <div className="input-group">
                    <input
                        className="days form-control"
                        type="number"
                        value={days}
                        placeholder="Days"
                        onChange={this.handleDaysChange}
                        onBlur={this.handleBlur}
                    />
                    <span className="input-group-addon" id="duration-days-unit">days</span>
                </div>
                <div className="input-group">
                    <input
                        className="hours form-control"
                        type="number"
                        value={hours}
                        placeholder="Hours"
                        onChange={this.handleHoursChange}
                        onBlur={this.handleBlur}
                    />
                    <span className="input-group-addon" id="duration-hours-unit">hours</span>
                </div>
                <div className="input-group">
                    <input
                        className="minutes form-control"
                        type="number"
                        value={minutes}
                        placeholder="Minutes"
                        onChange={this.handleMinutesChange}
                        onBlur={this.handleBlur}
                    />
                    <span className="input-group-addon" id="duration-minutes-unit">minutes</span>
                </div>
                {this.props.error && <span className="help-block">{this.props.error}</span>}
            </div>
        );
    }
}

DurationInput.propTypes = {
    error: PropTypes.string,
    duration: PropTypes.number,
    onChange: PropTypes.func,
};

DurationInput.defaultProps = {
    error: null,
    duration: 0,
    onChange: null,
};
