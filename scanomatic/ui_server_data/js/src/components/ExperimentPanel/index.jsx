import PropTypes from 'prop-types';
import React from 'react';
import ScanningJobStatusLabel from '../ScanningJobStatusLabel';
import { renderDuration } from '../ScanningJobPanelBody';
import Duration from '../../Duration';
import myProps from '../../prop-types';
import ScanningJobRemoveDialogue from '../ScanningJobRemoveDialogue';

const millisecondsPerMinute = 60000;

export default class ExperimentPanel extends React.Component {
    constructor(props) {
        super(props);
        this.state = { dialogue: null };
        this.handleDismissDialogue = () => this.setState({ dialogue: null });
        this.handleShowRemoveDialogue = () => this.setState({ dialogue: 'remove' });
    }

    render() {
        const {
            id, name, description, duration, interval, scanner, started, end,
            onStart, onRemove,
        } = this.props;
        const { dialogue } = this.state;
        const actions = [];
        const status = started ? 'Running' : 'Planned';

        if (status === 'Planned') {
            actions.push((
                <button
                    key="action-start"
                    type="button"
                    className="btn btn-default btn-block experiment-action-start"
                    onClick={() => onStart(id)}
                >
                    <span className="glyphicon glyphicon-play" /> Start
                </button>
            ));
            actions.push((
                <button
                    key="action-remove"
                    type="button"
                    className="btn btn-default btn-block experiment-action-remove"
                    onClick={this.handleShowRemoveDialogue}
                >
                    <span className="glyphicon glyphicon-remove" /> Remove
                </button>
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
                    <div className="row description-and-actions">
                        <div className="col-md-9">
                            <div className="text-justify experiment-description">{description}</div>
                        </div>
                        <div className="col-md-3 action-buttons">
                            {actions}
                        </div>
                    </div>
                </div>
                {dialogue && dialogue === 'remove' &&
                    <ScanningJobRemoveDialogue
                        name={name}
                        onConfirm={() => onRemove(id)}
                        onCancel={this.handleDismissDialogue}
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
                            {started &&
                                <tr className="experiment-started">
                                    <td>Started</td>
                                    <td>{started.toString()}</td>
                                </tr>
                            }
                            {end &&
                                <tr className="experiment-end">
                                    <td>{status === 'Running' ? 'Ends' : 'Ended'}</td>
                                    <td>{end.toString()}</td>
                                </tr>
                            }
                        </tbody>
                    </table>
                }
            </div>
        );
    }
}

ExperimentPanel.propTypes = {
    id: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    description: PropTypes.string,
    duration: PropTypes.number.isRequired,
    interval: PropTypes.number.isRequired,
    scanner: PropTypes.shape(myProps.scannerShape).isRequired,
    onStart: PropTypes.func.isRequired,
    onRemove: PropTypes.func.isRequired,
    started: PropTypes.instanceOf(Date),
    end: PropTypes.instanceOf(Date),
};

ExperimentPanel.defaultProps = {
    description: null,
    started: null,
    end: null,
};
