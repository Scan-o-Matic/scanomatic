import PropTypes from 'prop-types';
import React from 'react';

import NewScanningJob from '../components/NewScanningJob';
import { submitScanningJob } from '../api';
import SoMPropTypes from '../prop-types';

export default class NewScanningJobContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            duration: 3 * 24 * 3600 * 1000,
            interval: 20,
            scannerId: props.scanners.length > 0 ? props.scanners[0].identifier : '',
        };

        this.handleNameChange = this.handleNameChange.bind(this);
        this.handleDurationChange = this.handleDurationChange.bind(this);
        this.handleIntervalChange = this.handleIntervalChange.bind(this);
        this.handleSumbit = this.handleSumbit.bind(this);
        this.handleScannerChange = this.handleScannerChange.bind(this);
    }

    componentWillReceiveProps(nextProps) {
        if (this.state.scannerId === '' && nextProps.scanners.length > 0) {
            this.setState({ scannerId: nextProps.scanners[0].identifier });
        }
    }

    handleScannerChange(e) {
        this.setState({ scannerId: e.target.value });
    }

    handleNameChange(e) {
        this.setState({ name: e.target.value });
    }

    handleDurationChange(duration) {
        this.setState({ duration });
    }

    handleIntervalChange(e) {
        let minutes = Number(Number(e.target.value));
        if (minutes < 5) {
            minutes = 5;
        }
        this.setState({ interval: minutes });
    }

    handleSumbit() {
        submitScanningJob({
            name: this.state.name,
            duration: this.state.duration,
            interval: this.state.interval,
            scannerId: this.state.scannerId,
        })
            .then(this.props.onClose)
            .catch(reason => this.setState({ error: `Error submitting job: ${reason}` }));
    }

    render() {
        return (<NewScanningJob
            scanners={this.props.scanners}
            {...this.state}
            onSubmit={this.handleSumbit}
            onCancel={this.props.onClose}
            onIntervalChange={this.handleIntervalChange}
            onDurationChange={this.handleDurationChange}
            onNameChange={this.handleNameChange}
            onScannerChange={this.handleScannerChange}
        />);
    }
}

NewScanningJobContainer.propTypes = {
    onClose: PropTypes.func.isRequired,
    scanners: PropTypes.arrayOf(SoMPropTypes.scannerType).isRequired,
};
