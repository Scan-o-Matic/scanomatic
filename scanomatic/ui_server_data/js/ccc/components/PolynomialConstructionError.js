import PropTypes from 'prop-types';
import React from 'react';

export default function PolynomialConstructionError({ onClearError, error }) {
    return (
        <div className="alert alert-error alert-dismissible" role="alert">
            <button
                type="button"
                className="close"
                aria-label="Close"
                onClick={onClearError}
            >
                <span aria-hidden="true">&times;</span>
            </button>
            <span
                className="glyphicon glyphicon-exclamation-sign"
                aria-hidden="true"
            >
            </span>
            <strong>Error:</strong>
            {error}
        </div>
    );
}

PolynomialConstructionError.propTypes = {
    onClearError: PropTypes.func.isRequired,
    error: PropTypes.string.isRequired,
};
