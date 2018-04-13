import PropTypes from 'prop-types';
import React from 'react';


class ScanningJobStopDialogue extends React.Component {
    constructor() {
        super();
        this.state = { reason: '' };
        this.handleConfirm = this.handleConfirm.bind(this);
        this.handleReasonChange = this.handleReasonChange.bind(this);
    }

    handleConfirm() {
        this.props.onConfirm(this.state.reason);
    }

    handleReasonChange(e) {
        this.setState({ reason: e.target.value });
    }

    render() {
        const { name, onCancel } = this.props;
        const { reason } = this.state;
        return (
            <div
                className="jumbotron scanning-job-stop-dialogue"
            >
                <h1>Stop Job?</h1>
                <p>
                    This will stop the job {name} and free up the scanner with no
                    ability to undo this action. Analysis can be performed like
                    normal.
                </p>
                <input
                    className="reason-input"
                    value={reason}
                    placeholder="Reason"
                    onChange={this.handleReasonChange}
                />
                <p>
                    <button
                        className="btn btn-primary btn-lg confirm-button"
                        onClick={this.handleConfirm}
                    >
                        Yes
                    </button>
                    <button
                        className="btn btn-default btn-lg cancel-button"
                        onClick={onCancel}
                    >
                        No
                    </button>
                </p>
            </div>
        );
    }
}

ScanningJobStopDialogue.propTypes = {
    name: PropTypes.string.isRequired,
    onCancel: PropTypes.func.isRequired,
    onConfirm: PropTypes.func.isRequired,
};

export default ScanningJobStopDialogue;
