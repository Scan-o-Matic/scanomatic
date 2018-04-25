import PropTypes from 'prop-types';
import React from 'react';
import Duration from '../Duration';

export default class DurationInput extends React.Component {
    constructor(props) {
        super(props);
        this.handleDaysChange = this.handleDaysChange.bind(this);
        this.handleHoursChange = this.handleHoursChange.bind(this);
        this.handleMinutesChange = this.handleMinutesChange.bind(this);
        this.handleDaysFocus = () => this.setState({ daysFocus: true });
        this.handleHoursFocus = () => this.setState({ hoursFocus: true });
        this.handleMinutesFocus = () => this.setState({ minutesFocus: true });
        this.handleDaysBlur = this.handleDaysBlur.bind(this);
        this.handleHoursBlur = this.handleHoursBlur.bind(this);
        this.handleMinutesBlur = this.handleMinutesBlur.bind(this);
        this.state = {};
    }

    handleDaysChange(evt) {
        let val = Number.parseInt(evt.target.value, 10);
        if (Number.isNaN(val)) val = 0;
        if (this.state.daysFocus) {
            this.setState({ days: val });
        } else {
            this.handleNewDuration({ ...this.state, days: val });
        }
    }

    handleHoursChange(evt) {
        let val = Number.parseInt(evt.target.value, 10);
        if (Number.isNaN(val)) val = 0;
        if (this.state.hoursFocus) {
            this.setState({ hours: val });
        } else {
            this.handleNewDuration({ ...this.state, hours: val });
        }
    }

    handleMinutesChange(evt) {
        let val = Number.parseInt(evt.target.value, 10);
        if (Number.isNaN(val)) val = 0;
        if (this.state.minutesFocus) {
            this.setState({ minutes: val });
        } else {
            this.handleNewDuration({ ...this.state, minutes: val });
        }
    }

    handleDaysBlur() {
        this.setState({ daysFocus: false });
        this.handleNewDuration(this.state);
    }

    handleHoursBlur() {
        this.setState({ hoursFocus: false });
        this.handleNewDuration(this.state);
    }

    handleMinutesBlur() {
        this.setState({ minutesFocus: false });
        this.handleNewDuration(this.state);
    }

    getDisplayedValues(stateDuration) {
        let { days, hours, minutes } = stateDuration;
        const duration = Duration.fromMilliseconds(Math.max(0, this.props.duration));
        if (days == null) ({ days } = duration);
        if (hours == null) ({ hours } = duration);
        if (minutes == null) ({ minutes } = duration);
        return { days, hours, minutes };
    }

    handleNewDuration(stateDuration) {
        const { days, hours, minutes } = this.getDisplayedValues(stateDuration);
        const newDuration = Duration.fromDaysHoursMinutes(days || 0, hours || 0, minutes || 0);
        const value = newDuration.totalMilliseconds;
        this.setState({ days: null, hours: null, minutes: null });
        if (value !== this.props.duration) {
            if (this.props.onChange) this.props.onChange(value);
        }
    }

    render() {
        let { days, hours, minutes } = this.getDisplayedValues(this.state);
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
                        onBlur={this.handleDaysBlur}
                        onFocus={this.handleDaysFocus}
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
                        onBlur={this.handleHoursBlur}
                        onFocus={this.handleHoursFocus}
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
                        onBlur={this.handleMinutesBlur}
                        onFocus={this.handleMinutesFocus}
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
