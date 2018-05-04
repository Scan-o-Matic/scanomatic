import React from 'react';
import PropTypes from 'prop-types';

class ProjectPanel extends React.Component {
    constructor(props) {
        super(props);
        this.state = { expanded: false };
        this.handleToggleExpand = this.handleToggleExpand.bind(this);
    }

    handleToggleExpand() {
        this.setState({ expanded: !this.state.expanded });
    }

    render() {
        const {
            children,
            description,
            name,
            newExperimentDisabled,
            onNewExperiment,
        } = this.props;

        return (
            <div
                className="panel panel-default project-listing"
                data-projectname={name}
            >
                <div
                    className="panel-heading"
                    onClick={this.handleToggleExpand}
                    role="button"
                    tabIndex="-1"
                >
                    <div
                        className={
                            this.state.expanded
                                ? 'glyphicon glyphicon-collapse-up'
                                : 'glyphicon glyphicon-collapse-down'
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
                                    disabled={newExperimentDisabled}
                                >
                                    <div className="glyphicon glyphicon-plus" /> New Experiment
                                </button>
                            </div>
                        </div>
                        {children}
                    </div>
                )}
            </div>
        );
    }
}

ProjectPanel.propTypes = {
    children: PropTypes.node,
    description: PropTypes.string.isRequired,
    name: PropTypes.string.isRequired,
    newExperimentDisabled: PropTypes.bool,
    onNewExperiment: PropTypes.func.isRequired,
};

ProjectPanel.defaultProps = {
    newExperimentDisabled: false,
    children: null,
};

export default ProjectPanel;
