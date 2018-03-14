
import PropTypes from 'prop-types';
import React from 'react';

import SoMPropTypes from '../prop-types';

export function duration2milliseconds(duration) {
    if (duration) {
        return (duration.minutes * 60000) + (duration.hours * 3600000) + (duration.days * 86400000);
    }
    return 0;
}

export function getProgress(job) {
    const duration = duration2milliseconds(job.duration);
    const progress = new Date() - new Date(job.startTime);
    return Math.min(1, progress / duration) * 100;
}

export default function ScanningJobPanel(props) {
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
    let labelStyle = 'label label-default';
    if (props.status === 'Running') {
        labelStyle = 'label label-info';
    } else if (props.status === 'Completed') {
        labelStyle = 'label label-success';
    }
    return (
        <div className="panel panel-default job-listing" id={`job-${props.identifier}`}>
            <div className="panel-heading">
                <h3 className="panel-title">{props.name}</h3>
                <span className={labelStyle}>{props.status}</span>
            </div>
            {showStart}
            <div className="job-description">
                {status}
                <table className="table job-stats">
                    <tbody>
                        {jobDuration}
                        {scanFrequency}
                        {jobScanner}
                        {jobStart}
                        {jobEnd}
                    </tbody>
                </table>
            </div>
            {links}
        </div>
    );
}

ScanningJobPanel.propTypes = {
    duration: PropTypes.shape({
        days: PropTypes.number.isRequired,
        hours: PropTypes.number.isRequired,
        minutes: PropTypes.number.isRequired,
    }).isRequired,
    scanner: SoMPropTypes.scannerType,
    status: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    interval: PropTypes.number.isRequired,
    startTime: PropTypes.instanceOf(Date),
    endTime: PropTypes.instanceOf(Date),
    onStartJob: PropTypes.func.isRequired,
    disableStart: PropTypes.bool,
    identifier: PropTypes.string.isRequired,
};

ScanningJobPanel.defaultProps = {
    scanner: null,
    startTime: null,
    endTime: null,
    disableStart: false,
};
