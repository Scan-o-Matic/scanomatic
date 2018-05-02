import React from 'react';
import PropTypes from 'prop-types';
import myTypes from '../prop-types';
import NewExperimentPanel from './NewExperimentPanel';

class ProjectPanel extends React.Component {
    constructor(props) {
        super(props);
        this.state = { expanded: false };
        this.handleToggleExpand = this.handleToggleExpand.bind(this);
    }

    handleToggleExpand() {
        this.setState({ expanded: !this.state.expanded });
    }

    render () {
        const {
            name, description, onNewExperiment, newExperiment, newExperimentActions,
            newExperimentErrors, scanners,
        } = this.props;
        let newExperimentPanel;
        if (this.newExperiment) {
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
                <div className="panel-heading" onClick={this.handleToggleExpand} role="button">
                    <div
                        className={
                            this.state.expanded ?
                                'glyphicon glyphicon-collapse-up' :
                                'glyphicon glyphicon-collapse-down'
                        }
                    /> <h3 className="panel-title">{name}</h3>
                </div>
                {this.state.expanded && (
                    <div className="panel-body">
                        <div className="row">
                            <div className="col-md-9">
                                <div className="text-justify project-description">
                                    {description}
                                </div>
                            </div>
                            <div className="col-md-3 text-right">
                                <button
                                    className="btn btn-default new-experiment"
                                    onClick={() => onNewExperiment(name)}
                                    disabled={newExperiment}
                                >
                                    <div className="glyphicon glyphicon-plus" /> New Experiment
                                </button>
                            </div>
                        </div>
                        {newExperimentPanel}
                    </div>
                )}
            </div>
        );
    }
}

ProjectPanel.propTypes = {
    onNewExperiment: PropTypes.func.isRequired,
    newExperiment: PropTypes.shape(myTypes.experimentShape),
    newExperimentActions: PropTypes.shape({
        onChange: PropTypes.func,
        onCancel: PropTypes.func,
        onSubmit: PropTypes.func,
    }),
    newExperimentErrors: PropTypes.shape(myTypes.newExperimentErrorsShape),
    ...myTypes.projectShape,
};

ProjectPanel.defaultProps = {
    newExperimentErrors: null,
    newExperiment: null,
    newExperimentActions: {},
};

export default ProjectPanel;
