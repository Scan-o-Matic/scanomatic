import PropTypes from 'prop-types';
import React from 'react';

export default function NewProjectPanel({
    name, description, error, onChange, onSubmit, onCancel,
}) {
    return (
        <div className="panel panel-default new-project-panel">
            <div className="panel-heading">New project</div>
            {error && (
                <div className="alert alert-danger" role="alert">
                    {error}
                </div>
            )}
            <div className="panel-body">
                <form onSubmit={(event) => { event.preventDefault(); onSubmit(); }}>
                    <div className="form-group">
                        <label>Name</label>
                        <input
                            className="form-control name"
                            value={name}
                            onChange={event => onChange('name', event.target.value)}
                        />
                    </div>
                    <div className="form-group">
                        <label>Description</label>
                        <textarea
                            value={description}
                            onChange={event => onChange('description', event.target.value)}
                            className="form-control description"
                            rows="3"
                        />
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
    error: PropTypes.string,
    onCancel: PropTypes.func,
    onChange: PropTypes.func,
    onSubmit: PropTypes.func,
};

NewProjectPanel.defaultProps = {
    error: null,
    onCancel: null,
    onChange: null,
    onSubmit: null,
};
