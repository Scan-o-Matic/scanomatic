import PropTypes from 'prop-types';
import React from 'react';

export default function ProjectPanel({ name, description }) {
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
                    <div className="col-md-10">{description}</div>
                    <div className="col-md-2">
                        <button className="btn"><div className="glyphicon glyphicon-plus" /> New Experiment</button>
                    </div>
                </div>
            </div>
        </div>
    );
}

ProjectPanel.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
};
