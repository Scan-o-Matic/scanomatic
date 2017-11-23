import PropTypes from 'prop-types';
import React from 'react';

import CCCPropTypes from '../prop-types';
import CCCInitializationContainer from '../containers/CCCInitializationContainer';
import CCCEditorContainer from '../containers/CCCEditorContainer';


export default function Root(props) {
    let view;
    if (props.cccMetadata) {
        view = (
            <CCCEditorContainer
                cccMetadata={props.cccMetadata}
            />
        );
    } else {
        view = (
            <CCCInitializationContainer
                onError={props.onError}
                onInitialize={props.onInitializeCCC}
            />
        );
    }
    return (
        <div>
            {props.error && (
                <div className="alert alert-danger" role="alert">
                    {props.error}
                </div>
            )}
            {view}
        </div>
    );
}

Root.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata,
    error: PropTypes.string,
    onError: PropTypes.func.isRequired,
    onInitializeCCC: PropTypes.func.isRequired,
};

Root.defaultProps = {
    cccMetadata: null,
    error: null,
};
