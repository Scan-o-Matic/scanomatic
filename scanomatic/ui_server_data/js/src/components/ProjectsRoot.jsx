import React from 'react';
import PropTypes from 'prop-types';

import myTypes from '../prop-types';
import ProjectPanel from './ProjectPanel';
import ExperimentPanel from './ExperimentPanel';
import NewProjectPanel from './NewProjectPanel';
import NewExperimentPanel from './NewExperimentPanel';

export default class ProjectsRoot extends React.Component {
    renderProject(project) {
        const {
            onNewExperiment, newExperiment, newExperimentErrors, newExperimentActions,
            scanners, experimentActions,
        } = this.props;
        const hasNewExperiment = newExperiment && newExperiment.projectId === project.id;
        return (
            <ProjectPanel
                {...project}
                key={project.name}
                onNewExperiment={() => onNewExperiment(project.id)}
                newExperimentDisabled={hasNewExperiment}
                scanners={scanners}
            >
                {hasNewExperiment &&
                <NewExperimentPanel
                    {...newExperiment}
                    {...newExperimentActions}
                    projectName={project.name}
                    errors={newExperimentErrors}
                    scanners={scanners}
                />}
                {project.experiments.map(experiment => (
                    <ExperimentPanel key={experiment.id} {...experiment} {...experimentActions} />
                ))}
            </ProjectPanel>);
    }

    render() {
        const {
            projects, newProject, newProjectActions, newProjectErrors, onNewProject,
        } = this.props;
        let newProjectForm = null;
        if (newProject) {
            newProjectForm = (<NewProjectPanel
                {...newProject}
                {...newProjectActions}
                errors={newProjectErrors}
            />);
        }
        const newProjectButton = (
            <button className="btn btn-primary new-project" onClick={onNewProject} disabled={newProject}>
                <div className="glyphicon glyphicon-plus" /> New Project
            </button>
        );

        return (
            <div>
                <h1>Projects</h1>
                {newProjectButton}
                {newProjectForm}
                {projects.map(p => this.renderProject(p))}
            </div>
        );
    }
}

ProjectsRoot.propTypes = {
    newExperiment: PropTypes.shape(myTypes.experimentShape),
    newExperimentActions: PropTypes.shape({
        onChange: PropTypes.func.isRequired,
        onCancel: PropTypes.func.isRequired,
        onSubmit: PropTypes.func.isRequired,
    }).isRequired,
    newExperimentErrors: PropTypes.instanceOf(Map),
    newProject: PropTypes.shape(myTypes.projectShape),
    newProjectActions: PropTypes.shape({
        onChange: PropTypes.func.isRequired,
        onCancel: PropTypes.func.isRequired,
        onSubmit: PropTypes.func.isRequired,
    }).isRequired,
    experimentActions: PropTypes.shape({
        onStart: PropTypes.func.isRequired,
        onRemove: PropTypes.func.isRequired,
        onDialogue: PropTypes.func.isRequired,
    }).isRequired,
    newProjectErrors: PropTypes.instanceOf(Map),
    onNewExperiment: PropTypes.func.isRequired,
    onNewProject: PropTypes.func.isRequired,
    projects: PropTypes.arrayOf(PropTypes.shape(myTypes.projectShape)),
    scanners: PropTypes.arrayOf(PropTypes.shape(myTypes.scannerShape)),
};

ProjectsRoot.defaultProps = {
    newExperiment: undefined,
    newExperimentErrors: undefined,
    newProject: null,
    newProjectErrors: null,
    projects: [],
    scanners: [],
};
