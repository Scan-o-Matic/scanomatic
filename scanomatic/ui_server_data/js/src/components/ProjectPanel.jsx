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
                {description}
            </div>
        </div>
    );
}

ProjectPanel.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
};
