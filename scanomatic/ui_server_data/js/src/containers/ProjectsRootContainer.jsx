import { connect } from 'react-redux';
import ProjectRoot from '../components/ProjectsRoot';
import { initNewProject, clearNewProject, changeNewProject, submitNewProject } from '../projects/actions';
import { getNewProject, getNewProjectErrors, getProjects } from '../projects/selectors';

function mapStateToProps(state) {
    return {
        newProject: getNewProject(state),
        newProjectErrors: getNewProjectErrors(state),
        projects: getProjects(state),
    };
}


function mapDispatchToProps(dispatch) {
    return {
        onNewProject: () => dispatch(initNewProject()),
        newProjectActions: {
            onCancel: () => dispatch(clearNewProject()),
            onChange: (field, value) => dispatch(changeNewProject(field, value)),
            onSubmit: () => dispatch(submitNewProject()),
        },
    };
}

export default connect(mapStateToProps, mapDispatchToProps)(ProjectRoot);
