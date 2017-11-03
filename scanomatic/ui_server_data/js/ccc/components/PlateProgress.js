import PropTypes from 'prop-types';
import React from 'react';


export default function PlateProgress(props) {
    const width = `${Math.round(props.now / props.max * 100)}%`
    return (
        <div className="progress">
            <div
                className="progress-bar"
                role="progressbar"
                ariaValuenow={props.now}
                ariaValuemin={0}
                ariaValuemax={props.max}
                style={ { width, minWidth: '3em' } }
            >{props.now}/{props.max}</div>
        </div>
    );
}

PlateProgress.propTypes = {
    now: PropTypes.number.isRequired,
    max: PropTypes.number.isRequired,
};
