import PropTypes from 'prop-types';
import React from 'react';
import NewJobContainer from '../containers/NewJobContainer';
import JobList from './JobList';
import { jobType } from '../prop-types';

export default function Root(props) {
    const newJob = props.newJob ? <NewJobContainer onClose={props.onCloseNewJob} /> : null;
    return (
        <div className="row">
            {props.error && (
                <div className="alert alert-danger" role="alert">
                    {props.error}
                </div>
            )}
            <div className="col-md-6">
                <button
                    className="btn btn-primary btn-next"
                    onClick={props.onNewJob}
                    disabled={props.newJob}
                >
                    + Job
                </button>
                <JobList jobs={props.jobs} />
            </div>
            {newJob}
        </div>
    );
}

Root.propTypes = {
    error: PropTypes.string,
    newJob: PropTypes.bool.isRequired,
    onNewJob: PropTypes.func.isRequired,
    onCloseNewJob: PropTypes.func.isRequired,
    jobs: PropTypes.arrayOf(jobType).isRequired,
};

Root.defaultProps = {
    error: null,
};
