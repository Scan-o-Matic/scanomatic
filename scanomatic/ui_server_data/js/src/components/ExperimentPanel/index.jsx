import PropTypes from 'prop-types';
import React from 'react';
import ScanningJobStatusLabel from '../ScanningJobStatusLabel';
import { renderDuration } from '../ScanningJobPanelBody';
import Duration from '../../Duration';
import myProps from '../../prop-types';

const millisecondsPerMinute = 60000;

export default function ExperimentPanel({
    name, description, duration, interval, scanner, status,
}) {
    return (
        <div
            className="panel panel-default experiment-listing"
            data-experimentname={name}
        >
            <div className="panel-heading">
                <h3 className="panel-title">{name}</h3>
                <ScanningJobStatusLabel status={status} />
            </div>
            <div className="panel-body">
                <div className="row">
                    <div className="col-md-9">
                        <div className="text-justify experiment-description">{description}</div>
                    </div>
                    <div className="col-md-3" />
                </div>
            </div>
            <table className="table experiment-stats">
                <tbody>
                    <tr className="experiment-duration">
                        <td>Duration</td>
                        <td>{renderDuration(Duration.fromMilliseconds(duration))}</td>
                    </tr>
                    <tr className="experiment-interval">
                        <td>Interval</td>
                        <td>{Math.round(interval / millisecondsPerMinute)} minutes</td>
                    </tr>
                    <tr className="experiment-scanner">
                        <td>Scanner</td>
                        <td>{scanner.name} ({scanner.power ? 'online' : 'offline'}, {scanner.owned ? 'occupied' : 'free'})</td>
                    </tr>
                </tbody>
            </table>
        </div>
    );
}

ExperimentPanel.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    status: PropTypes.string,
    duration: PropTypes.number.isRequired,
    interval: PropTypes.number.isRequired,
    scanner: PropTypes.shape(myProps.scannerShape).isRequired,
};

ExperimentPanel.defaultProps = {
    description: null,
    status: 'Planned',
};
