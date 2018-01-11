
import PropTypes from 'prop-types';
import React from 'react';

import scannerType from '../prop-types';

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
    let showStart = null;
    if (props.scanner.owned || !props.scanner.power) {
        showStart = (
            <button type="button" className="btn btn-lg job-start" disabled>
                <span className="glyphicon glyphicon-ban-circle" /> Start
            </button>
        );
    } else {
        showStart = (
            <button type="button" className="btn btn-lg job-start">
                <span className="glyphicon glyphicon-play" /> Start
            </button>
        );
    }
    return (
        <div className="panel panel-default job-listing">
            <div className="panel-heading">
                <h3 className="panel-title">{props.name}</h3>
            </div>
            {showStart}
            <div className="job-description">
                Scan every {props.interval} minutes for {duration.join(' ')}.<br />
                Using scanner <b>{props.scanner.name}</b> (
                {props.scanner.power ? 'online' : 'offline'}, {props.scanner.owned ? 'occupied' : 'free'}).
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
    scanner: scannerType.isRequired,
    name: PropTypes.string.isRequired,
    interval: PropTypes.number.isRequired,
};
