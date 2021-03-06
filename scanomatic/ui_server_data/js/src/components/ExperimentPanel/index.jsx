import PropTypes from 'prop-types';
import React from 'react';
import ScanningJobStatusLabel from '../ScanningJobStatusLabel';
import { renderDuration } from '../ScanningJobPanelBody';
import Duration from '../../Duration';
import myProps from '../../prop-types';
import ScanningJobRemoveDialogue from '../ScanningJobRemoveDialogue';
import ScanningJobFeatureExtractDialogue from '../ScanningJobFeatureExtractDialogue';
import ScanningJobStopDialogue from '../ScanningJobStopDialogue';

const millisecondsPerMinute = 60000;

export function formatScannerStatus(scanner) {
    const { name, power, owned } = scanner;
    return `${name} (${power ? 'online' : 'offline'}, ${owned ? 'occupied' : 'free'})`;
}

export function formatPinning(pinning) {
    const spans = [];
    pinning.forEach((value, idx) => {
        const plate = idx + 1;
        if (value === '') {
            spans.push((
                <span className="pinning-format" key="plate-{plate}">
                    Plate {plate}: <span className="glyphicon glyphicon-ban-circle" aria-hidden="true" />
                </span>
            ));
        } else {
            spans.push((
                <span className="pinning-format" key="plate-{plate}">
                    Plate {plate}: <em>{value}</em>
                </span>
            ));
        }
    });
    return spans;
}

export default class ExperimentPanel extends React.Component {
    constructor(props) {
        super(props);
        this.state = { dialogue: null };
        this.handleDismissDialogue = () => this.setState({ dialogue: null });
        this.handleShowRemoveDialogue = () => this.setState({ dialogue: 'remove' });
        this.handleShowStopDialogue = () => this.setState({ dialogue: 'stop' });
        this.handleShowFeatureExtractDialogue = () => this.setState({ dialogue: 'extract' });
        this.handleToggleExpand = this.handleToggleExpand.bind(this);
        this.handleStopExperiment = (reason) => {
            this.setState({ dialogue: null });
            props.onStop(props.id, reason);
        };
    }

    getExpanded() {
        return this.state.expanded == null ? this.props.defaultExpanded : this.state.expanded;
    }

    getStatus() {
        const {
            started, end, stopped, done,
        } = this.props;
        if (done) return 'Done';
        if (!started) return 'Planned';
        if (stopped) return 'Analysis';
        if (end < new Date()) return 'Analysis';
        return 'Running';
    }

    getActionButtons(status) {
        const {
            id, name, onStart, onReopen, onDone,
        } = this.props;
        const actions = [];

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
        } else if (status === 'Running') {
            actions.push((
                <button
                    key="action-stop"
                    type="button"
                    className="btn btn-default btn-block experiment-action-stop"
                    onClick={this.handleShowStopDialogue}
                >
                    <span className="glyphicon glyphicon-stop" /> Stop
                </button>
            ));
        } else if (status === 'Analysis') {
            actions.push((
                <a
                    key="action-compile"
                    role="button"
                    className="btn btn-default btn-block experiment-action-compile"
                    href={`/compile?projectdirectory=root/${encodeURI(id)}`}
                >
                    <span className="glyphicon glyphicon-flash" /> Compile
                </a>
            ));
            actions.push((
                <a
                    key="action-analyse"
                    role="button"
                    className="btn btn-default btn-block experiment-action-analyse"
                    href={`/analysis?compilationfile=root/${encodeURI(id)}/${encodeURI(id)}.project.compilation`}
                >
                    <span className="glyphicon glyphicon-flash" /> Analyse
                </a>
            ));
            actions.push((
                <button
                    key="action-extract"
                    type="button"
                    className="btn btn-default btn-block experiment-action-extract"
                    onClick={this.handleShowFeatureExtractDialogue}
                >
                    <span className="glyphicon glyphicon-flash" /> Extract Features
                </button>
            ));
            actions.push((
                <a
                    key="action-qc"
                    role="button"
                    className="btn btn-default btn-block experiment-action-qc"
                    href={`/qc_norm?analysisdirectory=${encodeURI(id)}/analysis&project=${encodeURI(name)}`}
                >
                    <span className="glyphicon glyphicon-flash" /> Quality Control
                </a>
            ));
            actions.push((
                <button
                    key="action-done"
                    type="button"
                    className="btn btn-default btn-block experiment-action-done"
                    onClick={() => onDone(id)}
                >
                    <span className="glyphicon glyphicon-ok" /> Done
                </button>
            ));
        } else if (status === 'Done') {
            actions.push((
                <button
                    key="action-reopen"
                    type="button"
                    className="btn btn-default btn-block experiment-action-reopen"
                    onClick={() => onReopen(id)}
                >
                    <span className="glyphicon glyphicon-pencil" /> Re-open
                </button>
            ));
        }
        return actions;
    }

    getPanelContents(status) {
        const {
            id, name, description, duration, interval, scanner, started, end,
            onRemove, stopped, onFeatureExtract, pinning, reason,
        } = this.props;
        const contents = [];
        const { dialogue } = this.state;
        const actions = this.getActionButtons(status);

        contents.push(
            <div className="panel-body" key="panel-body">
                <div className="row description-and-actions">
                    <div className="col-md-9">
                        <div className="text-justify experiment-description description">
                            {description}
                        </div>
                    </div>
                    <div className="col-md-3 action-buttons">
                        {actions}
                    </div>
                </div>
            </div>,
        );
        if (dialogue === 'remove') {
            contents.push(
                <ScanningJobRemoveDialogue
                    name={name}
                    onConfirm={() => onRemove(id)}
                    onCancel={this.handleDismissDialogue}
                    key="remove-dialouge"
                />,
            );
        }
        if (dialogue === 'stop') {
            contents.push(
                <ScanningJobStopDialogue
                    name={name}
                    onConfirm={this.handleStopExperiment}
                    onCancel={this.handleDismissDialogue}
                    key="stop-dialouge"
                />,
            );
        }
        if (dialogue === 'extract') {
            contents.push(
                <ScanningJobFeatureExtractDialogue
                    name={name}
                    onConfirm={keepQC => onFeatureExtract(id, keepQC)}
                    onCancel={this.handleDismissDialogue}
                    key="extract-dialogue"
                />,
            );
        }
        if (!dialogue) {
            let endLabel = 'Ended';
            if (stopped) endLabel = 'Planned end';
            if (status === 'Running') endLabel = 'Ending';

            contents.push(
                <table className="table experiment-stats" key="experiment-stats">
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
                            <td>{formatScannerStatus(scanner)}</td>
                        </tr>
                        <tr className="experiment-pinning">
                            <td>Pinning</td>
                            <td>{formatPinning(pinning)}</td>
                        </tr>
                        {started &&
                            <tr className="experiment-started">
                                <td>Started</td>
                                <td>{started.toString()}</td>
                            </tr>
                        }
                        {end &&
                            <tr className="experiment-end">
                                <td>{endLabel}</td>
                                <td>{end.toString()}</td>
                            </tr>
                        }
                        {stopped &&
                            <tr className="experiment-stopped">
                                <td>Stopped</td>
                                <td>{stopped.toString()}</td>
                            </tr>
                        }
                        {reason &&
                            <tr className="experiment-stop-reason">
                                <td>Stop reason</td>
                                <td>{reason}</td>
                            </tr>
                        }
                    </tbody>
                </table>,
            );
        }

        return contents;
    }

    handleToggleExpand() {
        this.setState({ expanded: !this.getExpanded() });
    }

    render() {
        const { name } = this.props;
        const status = this.getStatus();
        const expanded = this.getExpanded();

        return (
            <div
                className="panel panel-default experiment-listing"
                data-experimentname={name}
            >
                <div
                    className="panel-heading"
                    onClick={this.handleToggleExpand}
                    role="button"
                    tabIndex="-1"
                >
                    <div
                        className={
                            expanded
                                ? 'glyphicon glyphicon-collapse-up'
                                : 'glyphicon glyphicon-collapse-down'
                        }
                    /> <h3 className="panel-title">{name}</h3>
                    <ScanningJobStatusLabel status={status} />
                </div>
                {expanded && this.getPanelContents(status)}
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
    pinning: PropTypes.arrayOf(PropTypes.string).isRequired,
    onStart: PropTypes.func.isRequired,
    onRemove: PropTypes.func.isRequired,
    onStop: PropTypes.func.isRequired,
    onDone: PropTypes.func.isRequired,
    onReopen: PropTypes.func.isRequired,
    onFeatureExtract: PropTypes.func.isRequired,
    started: PropTypes.instanceOf(Date),
    end: PropTypes.instanceOf(Date),
    stopped: PropTypes.instanceOf(Date),
    reason: PropTypes.string,
    done: PropTypes.bool,
    defaultExpanded: PropTypes.bool,
};

ExperimentPanel.defaultProps = {
    description: null,
    started: null,
    end: null,
    stopped: null,
    done: false,
    defaultExpanded: false,
    reason: null,
};
