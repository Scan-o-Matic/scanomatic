
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
    const jobDuration = (
        <tr>
            <td>Duration</td>
            <td>{duration.join(' ')}</td>
        </tr>
    );
    let scanFrequency;
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
            <tr>
                <td>Frequency</td>
                <td>Scanning every {props.interval} minutes</td>
            </tr>
        );
    } else if (props.status === 'Completed') {
        scanFrequency = (
            <tr>
                <td>Frequency</td>
                <td>Scanned every {props.interval} minutes</td>
            </tr>
        );
    } else if (props.disableStart || !scanner || scanner.owned || !scanner.power) {
        scanFrequency = (
            <tr>
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
            <tr>
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
    let jobScanner;
    if (props.status === 'Completed') {
        jobScanner = null;
    } else if (scanner) {
        jobScanner = (
            <tr>
                <td>Scanner</td>
                <td>{scanner.name} ({scanner.power ? 'online' : 'offline'}, {scanner.owned ? 'occupied' : 'free'})</td>
            </tr>
        );
    } else {
        jobScanner = (
            <tr>
                <td>Scanner</td>
                <td>Retrieving scanner status...</td>
            </tr>
        );
    }
    let jobStart;
    let jobEnd;
    if (props.startTime) {
        jobStart = (
            <tr>
                <td>Started</td>
                <td>{`${new Date(props.startTime)}`}</td>
            </tr>
        );
        if (props.status === 'Completed') {
            jobEnd = (
                <tr>
                    <td>Ended</td>
                    <td>{`${new Date(new Date(props.startTime) - -duration2milliseconds(props.duration))}`}</td>
                </tr>
            );
        } else {
            jobEnd = (
                <tr>
                    <td>Will end</td>
                    <td>{`${new Date(new Date(props.startTime) - -duration2milliseconds(props.duration))}`}</td>
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
            <div className="text-right">
                <a href={`/compile?projectdirectory=root/${props.identifier}`}>
                    Compile project
                </a>
                <a href={`/qc_norm?analysisdirectory=${encodeURI(props.identifier)}/analysis&project=${encodeURI(props.name)}`}>
                    QC project
                </a>
            </div>
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
    startTime: PropTypes.string,
    onStartJob: PropTypes.func.isRequired,
    disableStart: PropTypes.bool,
    identifier: PropTypes.string.isRequired,
};

ScanningJobPanel.defaultProps = {
    scanner: null,
    startTime: null,
    disableStart: false,
};
