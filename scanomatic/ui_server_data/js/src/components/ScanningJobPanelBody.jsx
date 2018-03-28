
import PropTypes from 'prop-types';
import React from 'react';

import SoMPropTypes from '../prop-types';
import ScanningJobRemoveButton from './ScanningJobRemoveButton';
import ScanningJobStopButton from './ScanningJobStopButton';

export function duration2milliseconds(duration) {
    if (duration) {
        const date = new Date(0);
        return date.setUTCHours((duration.days * 24) + duration.hours, duration.minutes);
    }
    return 0;
}

export function getProgress(job) {
    const duration = duration2milliseconds(job.duration);
    const progress = new Date() - new Date(job.startTime);
    return Math.min(1, progress / duration) * 100;
}

export default function ScanningJobPanelBody(props) {
    const duration = [];
    if (props.duration.days > 0) {
        duration.push(`${props.duration.days} days`);
    }
    if (props.duration.hours > 0) {
        duration.push(`${props.duration.hours} hours`);
    }
    if (props.duration.minutes > 0) {
        duration.push(`${props.duration.minutes} minutes`);
    }
    const { scanner } = props;
    let showStart = null;
    let status = null;
    let scanFrequency;
    let links;
    if (props.status === 'Running') {
        const progress = getProgress(props).toFixed(1);
        status = (
            <div className="progress">
                <div
                    className="progress-bar"
                    role="progressbar"
                    aria-valuenow={progress}
                    aria-valuemin="0"
                    aria-valuemax="100"
                    style={{ minWidth: '4em', width: `${progress}%` }}
                >
                    {progress}%
                </div>
            </div>
        );
        scanFrequency = (
            <tr className="job-info job-interval">
                <td>Frequency</td>
                <td>Scanning every {props.interval} minutes</td>
            </tr>
        );
    } else if (props.status === 'Completed') {
        scanFrequency = (
            <tr className="job-info job-interval">
                <td>Frequency</td>
                <td>Scanned every {props.interval} minutes</td>
            </tr>
        );
        links = (
            <div className="text-right job-links">
                <a href={`/compile?projectdirectory=root/${props.identifier}`}>
                    Compile project
                </a>
                <a href={`/qc_norm?analysisdirectory=${encodeURI(props.identifier)}/analysis&project=${encodeURI(props.name)}`}>
                    QC project
                </a>
            </div>
        );
    } else if (props.disableStart || !scanner || scanner.owned || !scanner.power) {
        scanFrequency = (
            <tr className="job-info job-interval">
                <td>Frequency</td>
                <td>Scan every {props.interval} minutes</td>
            </tr>
        );
        showStart = (
            <button type="button" className="btn btn-lg job-start" disabled>
                <span className="glyphicon glyphicon-ban-circle" /> Start
            </button>
        );
    } else {
        scanFrequency = (
            <tr className="job-info job-interval">
                <td>Frequency</td>
                <td>Scan every {props.interval} minutes</td>
            </tr>
        );
        showStart = (
            <button type="button" className="btn btn-lg job-start" onClick={props.onStartJob} >
                <span className="glyphicon glyphicon-play" /> Start
            </button>
        );
    }
    const jobDuration = (
        <tr className="job-info job-duration">
            <td>Duration</td>
            <td>{duration.join(' ')}</td>
        </tr>
    );
    let jobScanner;
    if (props.status === 'Completed') {
        jobScanner = null;
    } else if (scanner) {
        jobScanner = (
            <tr className="job-info job-scanner">
                <td>Scanner</td>
                <td>{scanner.name} ({scanner.power ? 'online' : 'offline'}, {scanner.owned ? 'occupied' : 'free'})</td>
            </tr>
        );
    } else {
        jobScanner = (
            <tr className="job-info job-scanner">
                <td>Scanner</td>
                <td>Retrieving scanner status...</td>
            </tr>
        );
    }
    let jobStart;
    let jobEnd;
    if (props.startTime) {
        jobStart = (
            <tr className="job-info job-start">
                <td>Started</td>
                <td>{`${props.startTime}`}</td>
            </tr>
        );
        if (props.status === 'Completed') {
            jobEnd = (
                <tr className="job-info job-end">
                    <td>Ended</td>
                    <td>{`${props.endTime}`}</td>
                </tr>
            );
        } else {
            jobEnd = (
                <tr className="job-info job-end">
                    <td>Will end</td>
                    <td>{`${props.endTime}`}</td>
                </tr>
            );
        }
    }
    return (
        <div>
            <div className="panel-body">
                {status}
                {showStart}
                {props.status === 'Planned' && <ScanningJobRemoveButton
                    identifier={props.identifier}
                    onRemoveJob={props.onRemoveJob}
                />}
                {props.status === 'Running' && <ScanningJobStopButton
                    onStopJob={props.onStopJob}
                />}
            </div>
            <table className="table job-stats">
                <tbody>
                    {jobDuration}
                    {scanFrequency}
                    {jobScanner}
                    {jobStart}
                    {jobEnd}
                </tbody>
            </table>
            {links}
        </div>
    );
}

ScanningJobPanelBody.propTypes = {
    disableStart: PropTypes.bool,
    duration: PropTypes.shape({
        days: PropTypes.number.isRequired,
        hours: PropTypes.number.isRequired,
        minutes: PropTypes.number.isRequired,
    }).isRequired,
    endTime: PropTypes.instanceOf(Date),
    identifier: PropTypes.string.isRequired,
    interval: PropTypes.number.isRequired,
    name: PropTypes.string.isRequired,
    onRemoveJob: PropTypes.func.isRequired,
    onStartJob: PropTypes.func.isRequired,
    onStopJob: PropTypes.func.isRequired,
    scanner: SoMPropTypes.scannerType,
    startTime: PropTypes.instanceOf(Date),
    status: PropTypes.string.isRequired,
};

ScanningJobPanelBody.defaultProps = {
    disableStart: false,
    endTime: null,
    scanner: null,
    startTime: null,
};
