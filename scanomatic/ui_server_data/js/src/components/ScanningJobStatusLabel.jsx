import PropTypes from 'prop-types';
import React from 'react';

const ScanningJobStatusLabel = (props) => {
    const { status } = props;
    const classNames = ['label', 'scanning-job-status-label'];
    if (status === 'Running') {
        classNames.push('label-info');
    } else if (status === 'Completed' || status === 'Done') {
        classNames.push('label-success');
    } else {
        classNames.push('label-default');
    }
    return <span className={classNames.join(' ')}>{status}</span>;
};

ScanningJobStatusLabel.propTypes = {
    status: PropTypes.oneOf(['Planned', 'Completed', 'Running', 'Analysis', 'Done']).isRequired,
};

export default ScanningJobStatusLabel;
