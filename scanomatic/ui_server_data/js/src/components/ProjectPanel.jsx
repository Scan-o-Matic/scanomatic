import React from 'react';
import PropTypes from 'prop-types';
import myTypes from '../prop-types';

export default function ProjectPanel({ name, description, onNewExperiment }) {
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
                        <button className="btn btn-default new-experiment" onClick={() => onNewExperiment(name)}><div className="glyphicon glyphicon-plus" /> New Experiment</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

ProjectPanel.propTypes = {
    onNewExperiment: PropTypes.func.isRequired,
    ...myTypes.projectShape,
};
