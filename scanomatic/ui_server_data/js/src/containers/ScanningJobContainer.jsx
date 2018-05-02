import React from 'react';
import PropTypes from 'prop-types';

import {
    startScanningJob,
    terminateScanningJob,
    extractFeatures,
} from '../api';
import ScanningJobPanel from '../components/ScanningJobPanel';
import SoMPropTypes from '../prop-types';


export default class ScanningJobContainer extends React.Component {
    constructor(props) {
        super(props);

        this.handleStartJob = this.handleStartJob.bind(this);
        this.handleStopJob = this.handleStopJob.bind(this);
        this.handleFeatureExtract = this.handleFeatureExtract.bind(this);
        this.handleErrorClose = this.handleErrorClose.bind(this);
        this.handleSuccessClose = this.handleSuccessClose.bind(this);
        this.state = {};
    }

    handleStartJob() {
        this.setState({ disableStart: true });
        return startScanningJob(this.props.scanningJob)
            .then(() => {
                this.props.updateFeed();
            })
            .catch((reason) => {
                this.setState({ error: `Error starting job: ${reason}`, disableStart: false });
            });
    }

    handleStopJob(jobId, reason) {
        return terminateScanningJob(jobId, reason)
            .then(() => {
                this.props.updateFeed();
            })
            .catch((message) => {
                this.setState({ error: `Error deleting job: ${message}` });
            });
    }

    handleErrorClose() {
        this.setState({ error: null });
    }

    handleSuccessClose() {
        this.setState({ successInfo: null });
    }

    handleFeatureExtract(keepQC) {
        return extractFeatures(this.props.scanningJob.identifier, 'analysis', keepQC)
            .then(() => this.setState({ successInfo: 'Feature extraction enqueued.' }))
            .catch((reason) => {
                if (reason) {
                    this.setState({ error: `Extraction refused: ${reason}` });
                } else {
                    this.setState({ error: 'Unexpected error: could not request feature extraction.' });
                }
            });
    }

    render() {
        const { error, successInfo, disableStart } = this.state;
        return (
            <ScanningJobPanel
                onStartJob={this.handleStartJob}
                onStopJob={this.handleStopJob}
                onFeatureExtract={this.handleFeatureExtract}
                onCloseError={this.handleErrorClose}
                onCloseSuccess={this.handleSuccessClose}
                error={error}
                successInfo={successInfo}
                disableStart={disableStart}
                {...this.props}
            />
        );
    }
}

ScanningJobContainer.propTypes = {
    scanningJob: PropTypes.shape(SoMPropTypes.scanningJobShape).isRequired,
    scanner: PropTypes.shape(SoMPropTypes.scannerShape),
    onRemoveJob: PropTypes.func.isRequired,
    updateFeed: PropTypes.func.isRequired,
};

ScanningJobContainer.defaultProps = {
    scanner: null,
};
