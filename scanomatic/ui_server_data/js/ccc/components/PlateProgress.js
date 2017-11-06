import PropTypes from 'prop-types';
import React from 'react';


export default function PlateProgress(props) {
    const width = `${Math.round(props.now / props.max * 100)}%`
    return (
        <div className="progress">
            <div
                className="progress-bar"
                role="progressbar"
                aria-valuenow={props.now}
                aria-valuemin={0}
                aria-valuemax={props.max}
                style={ { width, minWidth: '3em' } }
            >{props.now}/{props.max}</div>
        </div>
    );
}

PlateProgress.propTypes = {
    now: PropTypes.number.isRequired,
    max: PropTypes.number.isRequired,
};
