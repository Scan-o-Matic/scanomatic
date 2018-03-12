import PropTypes from 'prop-types';
import React from 'react';
import NewScanningJobContainer from '../containers/NewScanningJobContainer';
import SoMPropTypes from '../prop-types';
import ScanningJobPanel from './ScanningJobPanel';

export function duration2milliseconds(duration) {
    if (duration) {
        return (duration.minutes * 60000) + (duration.hours * 3600000) + (duration.days * 86400000);
    }
    return 0;
}

export function getStatus(startTime, duration, now) {
    const endTime = new Date(startTime) - -duration2milliseconds(duration);
    if (!startTime) {
        return 'Planned';
    } else if (endTime > now - 0) {
        return 'Running';
    } else if (endTime > now - 3600000) {
        return 'Completed';
    }
    return null;
}

export function jobSorter(a, b) {
    console.log(a, b);
    if (a.status === 'Planned' && b.status !== 'Planned') {
        return -1;
    } else if (b.status === 'Planned' && a.status !== 'Planned') {
        return 1;
    } else if (a.status === 'Planned') {
        return 0;
    } else if (a.status === 'Running' && b.status === 'Running') {
        return new Date(a.startTime) - new Date(b.startTime);
    } else if (a.status === 'Running') {
        return -1;
    } else if (b.status === 'Running') {
        return 1;
    }
    return (
        (new Date(b.startTime) - -duration2milliseconds(b.duration)) -
        (new Date(a.startTime) - -duration2milliseconds(a.duration))
    );
}

export default function ScanningRoot(props) {
    const { onStartJob } = props;
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
            .map(job => Object.assign({ status: getStatus(job.startTime, job.duration, now) }, job))
            .filter(job => job.status)
            .sort(jobSorter)
            .forEach((job) => {
                const scanner = props.scanners.filter(s => s.identifier === job.scannerId)[0];
                jobs.push(<ScanningJobPanel
                    onStartJob={() => onStartJob(job)}
                    key={job.name}
                    scanner={scanner}
                    {...job}
                />);
            });
        jobList = (
            <div className="jobs-list">
                {jobs.sort(jobSorter)}
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
    scanners: PropTypes.arrayOf(SoMPropTypes.scannerType).isRequired,
    onStartJob: PropTypes.func.isRequired,
};

ScanningRoot.defaultProps = {
    error: null,
};
