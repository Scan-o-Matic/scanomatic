import PropTypes from 'prop-types';
import React from 'react';

const ScanningJobStatusLabel = (props) => {
    const { status } = props;
    let className = 'label label-default';
    if (status === 'Running') {
        className = 'label label-info';
    } else if (status === 'Completed') {
        className = 'label label-success';
    }
    return <span className={className}>{status}</span>;
};

ScanningJobStatusLabel.propTypes = {
    status: PropTypes.oneOf(['Planned', 'Completed', 'Running']).isRequired,
};

export default ScanningJobStatusLabel;
