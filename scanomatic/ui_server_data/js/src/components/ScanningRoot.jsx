import PropTypes from 'prop-types';
import React from 'react';
import NewScanningJobContainer from '../containers/NewScanningJobContainer';
import ScanningJobsList from './ScanningJobsList';
import { scanningJobType } from '../prop-types';

export default function ScanningRoot(props) {
    const newJob = props.newJob ? <NewScanningJobContainer onClose={props.onCloseNewJob} /> : null;
    const jobList = newJob ? null : <ScanningJobsList jobs={props.jobs} />;
    return (
        <div className="row">
            {props.error && (
                <div className="alert alert-danger" role="alert">
                    {props.error}
                </div>
            )}
            <div className="col-md-6">
                <button
                    className="btn btn-primary btn-next new-job"
                    onClick={props.onNewJob}
                    disabled={props.newJob}
                >
                    <div className="glyphicon glyphicon-plus" /> Job
                </button>
                {jobList}
            </div>
            {newJob}
        </div>
    );
}

ScanningRoot.propTypes = {
    error: PropTypes.string,
    newJob: PropTypes.bool.isRequired,
    onNewJob: PropTypes.func.isRequired,
    onCloseNewJob: PropTypes.func.isRequired,
    jobs: PropTypes.arrayOf(scanningJobType).isRequired,
};

ScanningRoot.defaultProps = {
    error: null,
};
