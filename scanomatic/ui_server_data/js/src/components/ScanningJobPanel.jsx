
import PropTypes from 'prop-types';
import React from 'react';

import SoMPropTypes from '../prop-types';

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
    let scanVerb = 'Scan';
    if (props.disableStart || props.startTime) {
        scanVerb = 'Scanning';
    } else if (!scanner || scanner.owned || !scanner.power) {
        showStart = (
            <button type="button" className="btn btn-lg job-start" disabled>
                <span className="glyphicon glyphicon-ban-circle" /> Start
            </button>
        );
    } else {
        showStart = (
            <button type="button" className="btn btn-lg job-start" onClick={props.onStartJob} >
                <span className="glyphicon glyphicon-play" /> Start
            </button>
        );
    }
    let jobStatus = null;
    if (props.startTime) {
        jobStatus = (
            <div className="job-status">
                Started at {props.startTime}.
            </div>
        );
    }
    let scannerStatus = null;
    if (scanner) {
        scannerStatus = (
            <div className="scanner-status">
                Using scanner <b>{scanner.name}</b> ({scanner.power ? 'online' : 'offline'}, {scanner.owned ? 'occupied' : 'free'}).
            </div>
        );
    } else {
        scannerStatus = (
            <div className="scanner-status">
                Retrieving scanner status...
            </div>
        );
    }
    return (
        <div className="panel panel-default job-listing" id={`job-${props.identifier}`}>
            <div className="panel-heading">
                <h3 className="panel-title">{props.name}</h3>
            </div>
            {showStart}
            <div className="job-description">
                {scanVerb} every {props.interval} minutes for {duration.join(' ')}.
                {scannerStatus}
                {jobStatus}
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
