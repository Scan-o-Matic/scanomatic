import React from 'react';
import PropTypes from 'prop-types';

import myTypes from '../prop-types';
import ProjectPanel from './ProjectPanel';
import NewProjectPanel from './NewProjectPanel';

export default function ProjectsRoot({
    projects, newProject, newProjectActions, onNewProject,
}) {
    let newProjectForm = null;
    if (newProject) {
        newProjectForm = <NewProjectPanel {...newProject} {...newProjectActions} />;
    }
    const newProjectButton = (
        <button className="btn btn-primary new-project" onClick={onNewProject} disabled={newProject}>
            <div className="glyphicon glyphicon-plus" /> New Project
        </button>
    );

    const projectPanels = projects.map(project => <ProjectPanel {...project} key={project.name} />);
    return (
        <div>
            <h1>Projects</h1>
            {newProjectButton}
            {newProjectForm}
            {projectPanels}
        </div>
    );
}

ProjectsRoot.propTypes = {
    projects: PropTypes.arrayOf(myTypes.projectType),
    newProject: myTypes.projectType,
    newProjectActions: PropTypes.shape({
        error: PropTypes.string,
        onChange: PropTypes.func.isRequired,
        onCancel: PropTypes.func.isRequired,
        onSubmit: PropTypes.func.isRequired,
    }).isRequired,
    onNewProject: PropTypes.func.isRequired,
};

ProjectsRoot.defaultProps = {
    projects: [],
    newProject: null,
};
