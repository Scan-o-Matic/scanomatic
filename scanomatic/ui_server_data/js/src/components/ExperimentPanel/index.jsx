import PropTypes from 'prop-types';
import React from 'react';
import ScanningJobStatusLabel from '../ScanningJobStatusLabel';
import { renderDuration } from '../ScanningJobPanelBody';
import Duration from '../../Duration';
import myProps from '../../prop-types';
import ScanningJobRemoveButton from '../ScanningJobRemoveButton';
import ScanningJobRemoveDialogue from '../ScanningJobRemoveDialogue';

const millisecondsPerMinute = 60000;

export default function ExperimentPanel({
    id, name, description, duration, interval, scanner, status,
    onDialogue, onStart, onRemove, dialogue,
}) {
    const actions = [];
    if (status === 'Planned') {
        actions.push((
            <button key="action-start" type="button" className="btn btn-default btn-block experiment-action-start" onClick={onStart}>
                <span className="glyphicon glyphicon-play" /> Start
            </button>
        ));
        actions.push((
            <ScanningJobRemoveButton
                identifier={id}
                onRemoveJob={() => onDialogue(id, 'remove')}
                key="action-remove"
                className="btn btn-default btn-block experiment-action-remove"
            />
        ));
    }
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
                    <div className="col-md-3">
                        {actions}
                    </div>
                </div>
            </div>
            {dialogue && dialogue === 'remove' &&
                <ScanningJobRemoveDialogue
                    name={name}
                    onConfirm={() => onRemove(id)}
                    onCancel={() => onDialogue(id, null)}
                />
            }
            {!dialogue &&
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
            }
        </div>
    );
}

ExperimentPanel.propTypes = {
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    status: PropTypes.string,
    dialogue: PropTypes.string,
    duration: PropTypes.number.isRequired,
    interval: PropTypes.number.isRequired,
    scanner: PropTypes.shape(myProps.scannerShape).isRequired,
    onStart: PropTypes.func.isRequired,
    onDialogue: PropTypes.func.isRequired,
    onRemove: PropTypes.func.isRequired,
};

ExperimentPanel.defaultProps = {
    description: null,
    status: 'Planned',
    dialogue: null,
};
