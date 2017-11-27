import PropTypes from 'prop-types';
import React from 'react';

import CCCPropTypes from '../prop-types';
import CCCInitializationContainer from '../containers/CCCInitializationContainer';
import CCCEditorContainer from '../containers/CCCEditorContainer';
import FinalizedCCC from './FinalizedCCC';


export default function Root(props) {
    let view;
    if (props.cccMetadata && props.finalized) {
        view = <FinalizedCCC cccMetadata={props.cccMetadata} />;
    } else if (props.cccMetadata) {
        view = (
            <CCCEditorContainer
                cccMetadata={props.cccMetadata}
                onFinalizeCCC={props.onFinalizeCCC}
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
    finalized: PropTypes.bool,
    onError: PropTypes.func.isRequired,
    onFinalizeCCC: PropTypes.func.isRequired,
    onInitializeCCC: PropTypes.func.isRequired,
};

Root.defaultProps = {
    cccMetadata: null,
    error: null,
    finalized: false,
};
