
import PropTypes from 'prop-types';
import React from 'react';

export default function Job(props) {
    const duration = [];
    if (props.duration.days > 0) {
        duration.push(`${props.duration.days} days`);
    }
    if (props.duration.hours > 0) {
        duration.push(`${props.duration.hours} hours`);
    }
    if (props.duration.minutes > 0) {
        duration.push(`${props.duration.minutes} minutes`);
    }
    return (
        <div className="panel panel-default">
            <div className="panel-heading">
                <h3 className="panel-title">{props.name}</h3>
            </div>
            <button type="button" className="btn btn-lg">
                <span className="glyphicon glyphicon-play" /> Start
            </button>
            Scan every {props.interval} minutes for {duration.join(' ')}.
        </div>
    );
}

Job.propTypes = {
    duration: PropTypes.shape({
        days: PropTypes.number.isRequired,
        hours: PropTypes.number.isRequired,
        minutes: PropTypes.number.isRequired,
    }).isRequired,
    name: PropTypes.string.isRequired,
    interval: PropTypes.number.isRequired,
};
