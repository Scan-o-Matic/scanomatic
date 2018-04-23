import PropTypes from 'prop-types';
import React from 'react';

export default function NewProjectPanel({
    name, description, errors, onChange, onSubmit, onCancel,
}) {
    const nameError = errors.get('name');
    const descriptionError = errors.get('description');
    return (
        <div className="panel panel-default new-project-panel">
            <div className="panel-heading">New project</div>
            <div className="panel-body">
                <form onSubmit={(event) => { event.preventDefault(); onSubmit(); }}>
                    <div className={`form-group ${nameError ? 'has-error' : ''}`}>
                        <label className="control-label">Name</label>
                        <input
                            className="form-control name"
                            value={name}
                            onChange={event => onChange('name', event.target.value)}
                        />
                        {nameError && <span className="help-block">{nameError}</span>}
                    </div>
                    <div className={`form-group ${descriptionError ? 'has-error' : ''}`}>
                        <label className="control-label">Description</label>
                        <textarea
                            value={description}
                            onChange={event => onChange('description', event.target.value)}
                            className="form-control description"
                            rows="3"
                        />
                        {descriptionError && <span className="help-block">{descriptionError}</span>}
                    </div>
                    <button type="submit" className="btn btn-primary">
                        Add project
                    </button>
                    <button className="btn cancel" onClick={onCancel}>
                        Cancel
                    </button>
                </form>
            </div>
        </div>
    );
}

NewProjectPanel.propTypes = {
    name: PropTypes.string.isRequired,
    description: PropTypes.string.isRequired,
    errors: PropTypes.instanceOf(Map),
    onCancel: PropTypes.func,
    onChange: PropTypes.func,
    onSubmit: PropTypes.func,
};

NewProjectPanel.defaultProps = {
    errors: new Map(),
    onCancel: null,
    onChange: null,
    onSubmit: null,
};
