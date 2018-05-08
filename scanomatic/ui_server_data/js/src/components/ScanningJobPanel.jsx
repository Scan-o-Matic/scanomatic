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
        this.handleConfirmStop = this.handleConfirmStop.bind(this);
        this.handleShowFeatureExtractDialogue = this.handleShowFeatureExtractDialogue.bind(this);
        this.handleFeatureExtract = this.handleFeatureExtract.bind(this);
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

    handleShowFeatureExtractDialogue() {
        this.setState({ dialogue: 'featureExtact' });
    }

    handleFeatureExtract(keepQC) {
        this.setState({ dialogue: null });
        this.props.onFeatureExtract(keepQC);
    }

    render() {
        const {
            onStartJob,
            scanner,
            scanningJob,
            error,
            successInfo,
            onCloseError,
            onCloseSuccess,
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
                {error &&
                    <div className="alert alert-danger alert-dismissible" role="alert">
                        <button type="button" className="close" data-dismiss="alert" aria-label="Close" onClick={onCloseError}>
                            <span aria-hidden="true">&times;</span>
                        </button>
                        {error}
                    </div>
                }
                {successInfo &&
                    <div className="alert alert-success alert-dismissible" role="alert">
                        <button type="button" className="close" data-dismiss="alert" aria-label="Close" onClick={onCloseSuccess}>
                            <span aria-hidden="true">&times;</span>
                        </button>
                        {successInfo}
                    </div>
                }
                {!dialogue &&
                    <ScanningJobPanelBody
                        {...scanningJob}
                        onRemoveJob={this.handleRemove}
                        onStartJob={onStartJob}
                        onStopJob={this.handleStop}
                        onShowFeatureExtractDialogue={this.handleShowFeatureExtractDialogue}
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
                        onConfirm={this.handleFeatureExtract}
                    />
                }
            </div>
        );
    }
}

ScanningJobPanel.propTypes = {
    scanningJob: PropTypes.shape(SoMPropTypes.scanningJobShape).isRequired,
    scanner: PropTypes.shape(SoMPropTypes.scannerShape),
    onStartJob: PropTypes.func.isRequired,
    onRemoveJob: PropTypes.func.isRequired,
    onStopJob: PropTypes.func.isRequired,
    onFeatureExtract: PropTypes.func.isRequired,
    onCloseError: PropTypes.func.isRequired,
    onCloseSuccess: PropTypes.func.isRequired,
    error: PropTypes.string,
    successInfo: PropTypes.string,
};

ScanningJobPanel.defaultProps = {
    scanner: null,
    error: null,
    successInfo: null,
};

export default ScanningJobPanel;
