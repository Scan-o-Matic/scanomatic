import { connect } from 'react-redux';
import ProjectRoot from '../components/ProjectsRoot';
import {
    changeNewExperiment,
    changeNewProject,
    clearNewExperiment,
    clearNewProject,
    initNewExperiment,
    initNewProject,
    submitNewExperiment,
    submitNewProject,
    startExperiment,
    removeExperiment,
    stopExperiment,
    featureExtact,
    doneExperiment,
    reopenExperiment,
} from '../projects/actions';
import {
    getNewExperiment,
    getNewExperimentErrors,
    getNewProject,
    getNewProjectErrors,
    getProjects,
    getScanners,
} from '../projects/selectors';

function mapStateToProps(state) {
    return {
        newProject: getNewProject(state),
        newProjectErrors: getNewProjectErrors(state),
        newExperiment: getNewExperiment(state),
        newExperimentErrors: getNewExperimentErrors(state),
        projects: getProjects(state),
        scanners: getScanners(state),
    };
}


function mapDispatchToProps(dispatch) {
    return {
        onNewProject: () => dispatch(initNewProject()),
        onNewExperiment: projectId => dispatch(initNewExperiment(projectId)),
        newExperimentActions: {
            onCancel: () => dispatch(clearNewExperiment()),
            onChange: (field, value) => dispatch(changeNewExperiment(field, value)),
            onSubmit: () => dispatch(submitNewExperiment()),
        },
        newProjectActions: {
            onCancel: () => dispatch(clearNewProject()),
            onChange: (field, value) => dispatch(changeNewProject(field, value)),
            onSubmit: () => dispatch(submitNewProject()),
        },
        experimentActions: {
            onStart: experimentId => dispatch(startExperiment(experimentId)),
            onRemove: experimentId => dispatch(removeExperiment(experimentId)),
            onStop:
                (experimentId, reason) => dispatch(stopExperiment(experimentId, reason)),
            onFeatureExtact:
                (experimentId, keepQC) => dispatch(featureExtact(experimentId, keepQC)),
            onDone: experimentId => dispatch(doneExperiment(experimentId)),
            onReopen: experimentId => dispatch(reopenExperiment(experimentId)),
        },
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(ProjectRoot);
