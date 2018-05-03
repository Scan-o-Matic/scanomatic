import PropTypes from 'prop-types';
import React from 'react';
import NewScanningJobContainer from '../containers/NewScanningJobContainer';
import SoMPropTypes from '../prop-types';
import ScanningJobContainer from '../containers/ScanningJobContainer';

export function getStatus(job, now) {
    if (!job.startTime) {
        return 'Planned';
    } else if (job.terminationTime && job.terminationTime < now) {
        return 'Completed';
    } else if (job.duration.after(job.startTime) > now) {
        return 'Running';
    }
    return 'Completed';
}

export function jobSorter(a, b) {
    if (a.status === 'Planned' && b.status !== 'Planned') {
        return -1;
    } else if (b.status === 'Planned' && a.status !== 'Planned') {
        return 1;
    } else if (a.status === 'Planned') {
        return 0;
    } else if (a.status === 'Running' && b.status === 'Running') {
        return a.endTime - b.endTime;
    } else if (a.status === 'Running') {
        return -1;
    } else if (b.status === 'Running') {
        return 1;
    }
    return b.startTime - a.startTime;
}

export default function ScanningRoot(props) {
    let newJob = null;
    if (props.newJob) {
        newJob = (<NewScanningJobContainer
            onClose={props.onCloseNewJob}
            scanners={props.scanners}
        />);
    }
    let jobList = null;
    if (!newJob) {
        const jobs = [];
        const now = new Date();
        props.jobs
            .map(job => Object.assign({ status: getStatus(job, now) }, job))
            .filter(job => job.status)
            .sort(jobSorter)
            .forEach((job) => {
                const scanner = props.scanners.filter(s => s.identifier === job.scannerId)[0];
                jobs.push(<ScanningJobContainer
                    key={job.name}
                    onRemoveJob={props.onRemoveJob}
                    scanner={scanner}
                    scanningJob={job}
                    updateFeed={props.updateFeed}
                />);
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
            <div className="col-md-12">
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
    scanners: PropTypes.arrayOf(SoMPropTypes.scannerType).isRequired,
    onRemoveJob: PropTypes.func.isRequired,
    updateFeed: PropTypes.func.isRequired,
};

ScanningRoot.defaultProps = {
    error: null,
};
