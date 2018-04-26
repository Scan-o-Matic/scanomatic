import PropTypes from 'prop-types';
import React from 'react';

import SoMPropTypes from '../prop-types';
import ScanningJobPanelBody from './ScanningJobPanelBody';
import ScanningJobRemoveDialogue from './ScanningJobRemoveDialogue';
import ScanningJobStopDialogue from './ScanningJobStopDialogue';
import ScanningJobStatusLabel from './ScanningJobStatusLabel';
import ScanningJobFeatureExtractDialogue from './ScanningJobFeatureExtractDialogue';

class ScanningJobPanel extends React.Component {
    constructor() {
        super();
        this.state = { dialogue: null };
        this.handleRemove = this.handleRemove.bind(this);
        this.handleCancel = () => this.setState({ dialogue: null });
        this.handleConfirmRemove = this.handleConfirmRemove.bind(this);
        this.handleStop = this.handleStop.bind(this);
        this.handleCancelStop = this.handleCancelStop.bind(this);
        this.handleConfirmStop = this.handleConfirmStop.bind(this);
    }

    handleRemove() {
        this.setState({ dialogue: 'remove' });
    }

    handleConfirmRemove() {
        this.setState({ dialogue: null });
        this.props.onRemoveJob(this.props.scanningJob.identifier);
    }

    handleStop() {
        this.setState({ dialogue: 'stop' });
    }

    handleConfirmStop(reason) {
        this.setState({ dialogue: null });
        this.props.onStopJob(this.props.scanningJob.identifier, reason);
    }

    handleFeatureExtract(keepQC) {
        this.setState({ dialogue: null });
        this.props.onExtractFeatures(this.props.scanningJob.identifier, keepQC);
    }

    render() {
        const {
            onStartJob,
            scanner,
            scanningJob,
        } = this.props;
        const { identifier, name, status } = scanningJob;
        const { dialogue } = this.state;

        return (
            <div
                className="panel panel-default job-listing"
                id={`job-${identifier}`}
                data-jobname={name}
            >
                <div className="panel-heading">
                    <h3 className="panel-title">{name}</h3>
                    <ScanningJobStatusLabel status={status} />
                </div>
                {!dialogue &&
                    <ScanningJobPanelBody
                        {...scanningJob}
                        onRemoveJob={this.handleRemove}
                        onStartJob={onStartJob}
                        onStopJob={this.handleStop}
                        scanner={scanner}
                    />
                }
                {dialogue === 'remove' &&
                    <ScanningJobRemoveDialogue
                        name={name}
                        onCancel={this.handleCancel}
                        onConfirm={this.handleConfirmRemove}
                    />
                }
                {dialogue === 'stop' &&
                    <ScanningJobStopDialogue
                        name={name}
                        onCancel={this.handleCancel}
                        onConfirm={this.handleConfirmStop}
                    />
                }
                {dialogue === 'featureExtact' &&
                    <ScanningJobFeatureExtractDialogue
                        onCancel={this.handleCancel}
                        onExtractFeatures={this.handleExtractFeatures}
                    />
                }
            </div>
        );
    }
}

ScanningJobPanel.propTypes = {
    scanningJob: SoMPropTypes.scanningJobType.isRequired,
    scanner: SoMPropTypes.scannerType,
    onStartJob: PropTypes.func.isRequired,
    onRemoveJob: PropTypes.func.isRequired,
    onStopJob: PropTypes.func.isRequired,
    onExtractFeatures: PropTypes.func.isRequired,
};

export default ScanningJobPanel;
