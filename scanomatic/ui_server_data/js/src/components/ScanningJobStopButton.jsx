import PropTypes from 'prop-types';
import React from 'react';

const ScanningJobStopButton = (props) => {
    const { onStopJob } = props;
    return (
        <button
            type="button"
            className="btn btn-lg scanning-job-stop-button"
            onClick={onStopJob}
        >
            <span className="glyphicon glyphicon-remove" /> Stop
        </button>
    );
};

ScanningJobStopButton.propTypes = {
    onStopJob: PropTypes.func.isRequired,
};

export default ScanningJobStopButton;
