import PropTypes from 'prop-types';
import React from 'react';

const ScanningJobRemoveButton = (props) => {
    const { onRemoveJob, className } = props;
    return (
        <button
            type="button"
            className={className || 'btn btn-lg scanning-job-remove-button'}
            onClick={onRemoveJob}
        >
            <span className="glyphicon glyphicon-remove" /> Remove
        </button>
    );
};

ScanningJobRemoveButton.propTypes = {
    onRemoveJob: PropTypes.func.isRequired,
    className: PropTypes.string,
};

ScanningJobRemoveButton.defaultProps = {
    className: null,
};

export default ScanningJobRemoveButton;
