import PropTypes from 'prop-types';
import React from 'react';


const ScanningJobRemoveDialogue = (props) => {
    const { name, onConfirm, onCancel } = props;
    return (
        <div
            className="jumbotron scanning-job-remove-dialogue"
        >
            <h1>Remove Job?</h1>
            <p>
                This will permanently remove the planned job <em>{name}</em> with no
                ability to undo this action.
            </p>
            <p>
                <button
                    className="btn btn-primary btn-lg confirm-button"
                    onClick={onConfirm}
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
};

ScanningJobRemoveDialogue.propTypes = {
    name: PropTypes.string.isRequired,
    onCancel: PropTypes.func.isRequired,
    onConfirm: PropTypes.func.isRequired,
};

export default ScanningJobRemoveDialogue;
