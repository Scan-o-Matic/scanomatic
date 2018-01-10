import PropTypes from 'prop-types';
import React from 'react';

import NewJob from '../components/NewJob';
import { submitJob, getFreeScanners } from '../api'

export default class NewJobContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            duration: {
                days: 3,
                hours: 0,
                minutes: 0,
            },
            interval: 20,
        };

        this.handleNameChange = this.handleNameChange.bind(this);
        this.handleDurationDaysChange = this.handleDurationDaysChange.bind(this);
        this.handleDurationHoursChange = this.handleDurationHoursChange.bind(this);
        this.handleDurationMinutesChange = this.handleDurationMinutesChange.bind(this);
        this.handleIntervalChange = this.handleIntervalChange.bind(this);
        this.handleSumbit = this.handleSumbit.bind(this);
        this.handleScannerNameChange = this.handleScannerNameChange.bind(this);
    }

    componentDidMount() {
        getFreeScanners()
            .then((r) => {
                const selectedName = r.length > 0 ? r[0].name : '';
                this.setState({ scanners: r, scannerName: selectedName });
            })
            .catch(reason => this.setState({ error: `Error retrieving scanners: ${reason}` }));
    }

    handleScannerNameChange(e) {
        this.setState({ scannerName: e.target.value });
    }

    handleNameChange(e) {
        this.setState({ name: e.target.value });
    }

    handleDurationDaysChange(e) {
        let days = Number(Number(e.target.value).toFixed(0));
        if (days < 0) {
            days = 0;
        }
        const duration = Object.assign({}, this.state.duration);
        duration.days = days;
        this.setState({ duration });
    }

    handleDurationHoursChange(e) {
        let hours = Number(Number(e.target.value).toFixed(0));
        let { days } = this.state.duration;
        if (hours < 0) {
            hours = 0;
        } else if (hours > 23) {
            hours -= 24;
            days += 1;
        }
        const duration = Object.assign({}, this.state.duration);
        duration.days = days;
        duration.hours = hours;
        this.setState({ duration });
    }

    handleDurationMinutesChange(e) {
        let minutes = Number(Number(e.target.value).toFixed(0));
        let { days, hours } = this.state.duration;
        if (minutes < 0) {
            minutes = 0;
        } else if (minutes > 59) {
            minutes -= 60;
            hours += 1;
        }
        if (hours < 0) {
            hours = 0;
        } else if (hours > 23) {
            hours -= 24;
            days += 1;
        }
        this.setState({ duration: { days, hours, minutes } });
    }

    handleIntervalChange(e) {
        let minutes = Number(Number(e.target.value));
        if (minutes < 5) {
            minutes = 5;
        }
        this.setState({ interval: minutes });
    }

    handleSumbit() {
        submitJob({
            name: this.state.name,
            duration: this.state.duration,
            interval: this.state.interval,
            scannerName: this.state.scannerName,
        })
            .then(this.props.onClose)
            .catch(reason => this.setState({ error: `Error submitting job: ${reason}` }));
    }

    render() {
        return (<NewJob
            {...this.state}
            onSubmit={this.handleSumbit}
            onCancel={this.props.onClose}
            onIntervalChange={this.handleIntervalChange}
            onDurationMinutesChange={this.handleDurationMinutesChange}
            onDurationHoursChange={this.handleDurationHoursChange}
            onDurationDaysChange={this.handleDurationDaysChange}
            onNameChange={this.handleNameChange}
            onScannerNameChange={this.handleScannerNameChange}
        />);
    }
}

NewJobContainer.propTypes = {
    onClose: PropTypes.func.isRequired,
};
