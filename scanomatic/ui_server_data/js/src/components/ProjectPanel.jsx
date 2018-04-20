import React from 'react';
import PropTypes from 'prop-types';
import myTypes from '../prop-types';
import NewExperimentPanel from './NewExperimentPanel';

export default function ProjectPanel({
    name, description, onNewExperiment, newExperiment, newExperimentActions, newExperimentErrors,
    scanners,
}) {
    let newExperimentPanel;
    if (newExperiment) {
        newExperimentPanel = (
            <NewExperimentPanel
                errors={newExperimentErrors}
                scanners={scanners}
                {...newExperiment}
                {...newExperimentActions}
            />
        );
    }

    return (
        <div
            className="panel panel-default project-listing"
            data-projectname={name}
        >
            <div className="panel-heading">
                <h3 className="panel-title">{name}</h3>
            </div>
            <div className="panel-body">
                <div className="row">
                    <div className="col-md-9"><div className="text-justify project-description">{description}</div></div>
                    <div className="col-md-3 text-right">
                        <button className="btn btn-default new-experiment" onClick={() => onNewExperiment(name)} disabled={newExperiment}>
                            <div className="glyphicon glyphicon-plus" /> New Experiment
                        </button>
                    </div>
                </div>
                {newExperimentPanel}
            </div>
        </div>
    );
}

ProjectPanel.propTypes = {
    onNewExperiment: PropTypes.func.isRequired,
    newExperiment: PropTypes.shape(myTypes.experimentShape),
    newExperimentActions: PropTypes.shape({
        onNameChange: PropTypes.func.isRequired,
        onDescriptionChange: PropTypes.func.isRequired,
        onDurationDaysChange: PropTypes.func.isRequired,
        onDurationHoursChange: PropTypes.func.isRequired,
        onDurationMinutesChange: PropTypes.func.isRequired,
        onIntervalChange: PropTypes.func.isRequired,
        onCancel: PropTypes.func.isRequired,
        onSubmit: PropTypes.func.isRequired,
    }),
    newExperimentErrors: PropTypes.shape(myTypes.newExperimentErrorsShape),
    ...myTypes.projectShape,
};

ProjectPanel.defaultProps = {
    newExperimentErrors: null,
    newExperiment: null,
    newExperimentActions: {},
};
