import PropTypes from 'prop-types';
import React from 'react';

import SoMPropTypes from '../prop-types';
import ScanningJobPanelBody from './ScanningJobPanelBody';
import ScanningJobRemoveDialogue from './ScanningJobRemoveDialogue';
import ScanningJobStatusLabel from './ScanningJobStatusLabel';

class ScanningJobPanel extends React.Component {
    constructor() {
        super();
        this.state = { dialogue: null };
        this.handleRemove = this.handleRemove.bind(this);
        this.handleCancelRemove = this.handleCancelRemove.bind(this);
        this.handleConfirmRemove = this.handleConfirmRemove.bind(this);
    }

    handleRemove() {
        this.setState({ dialogue: 'remove' });
    }

    handleCancelRemove() {
        this.setState({ dialogue: null });
    }

    handleConfirmRemove() {
        this.setState({ dialogue: null });
        this.props.onRemoveJob(this.props.scanningJob.identifier);
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
                        scanner={scanner}
                    />
                }
                {dialogue === 'remove' &&
                    <ScanningJobRemoveDialogue
                        name={name}
                        onCancel={this.handleCancelRemove}
                        onConfirm={this.handleConfirmRemove}
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
};

export default ScanningJobPanel;
