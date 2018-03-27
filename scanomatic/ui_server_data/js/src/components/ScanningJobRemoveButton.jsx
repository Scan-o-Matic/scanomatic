import PropTypes from 'prop-types';
import React from 'react';

const ScanningJobRemoveButton = (props) => {
    const { onRemoveJob } = props;
    return (
        <button
            type="button"
            className="btn btn-lg scanning-job-remove-button"
            onClick={onRemoveJob}
        >
            <span className="glyphicon glyphicon-remove" /> Remove
        </button>
    );
};

ScanningJobRemoveButton.propTypes = {
    onRemoveJob: PropTypes.func.isRequired,
};

export default ScanningJobRemoveButton;
