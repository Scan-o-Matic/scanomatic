import PropTypes from 'prop-types';
import React from 'react';

import Job from './Job';
import { jobType } from '../prop-types';

export default function JobList(props) {
    const jobs = [];
    props.jobs.forEach((job) => {
        jobs.push(<Job key={job.name} {...job} />);
    });
    return (
        <div>
            {jobs}
        </div>
    );
}

JobList.propTypes = {
    jobs: PropTypes.arrayOf(jobType).isRequired,
};
