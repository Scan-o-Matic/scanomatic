import React from 'react';

import SoMPropTypes from '../prop-types';

export default function FinalizedCCC(props) {
    return (
        <div className="row">
            <div className="col-md-6 col-md-offset-3 text-center">
                <h3>Well Done!</h3>
                <p>
                    Calibration activated and ready to use.
                    It will be available as
                    &ldquo;{props.cccMetadata.species}, {props.cccMetadata.reference}&rdquo;
                    in new experiments.
                </p>
                <div className="text-center">
                    <a href="/home" className="btn btn-primary">Go to home page</a>
                </div>
            </div>
        </div>
    );
}

FinalizedCCC.propTypes = {
    cccMetadata: SoMPropTypes.cccMetadata.isRequired,
};
