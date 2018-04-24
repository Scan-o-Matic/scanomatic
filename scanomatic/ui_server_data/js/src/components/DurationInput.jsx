import PropTypes from 'prop-types';
import React from 'react';
import Duration from '../Duration';

export default class DurationInput extends React.Component {
    constructor(props) {
        super(props);
        this.handleDaysChange = this.handleDaysChange.bind(this);
        this.handleHoursChange = this.handleHoursChange.bind(this);
        this.handleMinutesChange = this.handleMinutesChange.bind(this);
        const duration = Duration.fromMilliseconds(Math.max(0, this.props.duration));
        this.state = {
            duration,
        };
    }
    handleDaysChange(evt) {
        let val = Number.parseInt(evt.target.value, 10);
        if (Number.isNaN(val)) val = 0;
        this.setState({ days: val });
    }

    handleHoursChange(evt) {
        let val = Number.parseInt(evt.target.value, 10);
        if (Number.isNaN(val)) val = 0;
        this.setState({ hours: val });
    }

    handleMinutesChange(evt) {
        let val = Number.parseInt(evt.target.value, 10);
        if (Number.isNaN(val)) val = 0;
        this.setState({ minutes: val });
    }

    handleNewDuration(duration) {
        const { days, hours, minutes } = (duration || {});
        const newDuration = Duration.fromDaysHoursMinutes(days || 0, hours || 0, minutes || 0);
        const value = newDuration.totalMilliseconds;
        if (value !== this.props.duration) {
            if (this.props.onChange) this.props.onChange(value);
        }
        this.setState({
            days: newDuration.days,
            hours: newDuration.hours,
            minutes: newDuration.minutes,
        });
    }

    render() {
        const { duration } = this.state;
        let { days, hours, minutes } = this.state;
        if (days == null) ({ days } = duration);
        if (hours == null) ({ hours } = duration);
        if (minutes == null) ({ minutes } = duration);
        const blurValues = { hours, minutes, days };
        const handleBlur = () => this.handleNewDuration(blurValues);
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
                        onBlur={handleBlur}
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
                        onBlur={handleBlur}
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
                        onBlur={handleBlur}
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
