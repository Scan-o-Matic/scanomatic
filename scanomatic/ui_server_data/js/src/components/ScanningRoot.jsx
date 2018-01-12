import PropTypes from 'prop-types';
import React from 'react';
import NewScanningJobContainer from '../containers/NewScanningJobContainer';
import SoMPropTypes from '../prop-types';
import ScanningJobPanel from './ScanningJobPanel';

export default function ScanningRoot(props) {
    const newJob = props.newJob ? <NewScanningJobContainer onClose={props.onCloseNewJob} /> : null;
    let jobList = null;
    if (!newJob) {
        const jobs = [];
        props.jobs.forEach((job) => {
            jobs.push(<ScanningJobPanel key={job.name} {...job} />);
        });
        jobList = (
            <div className="jobs-list">
                {jobs}
            </div>
        );
    }

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
    jobs: PropTypes.arrayOf(SoMPropTypes.scanningJobType).isRequired,
};

ScanningRoot.defaultProps = {
    error: null,
};
