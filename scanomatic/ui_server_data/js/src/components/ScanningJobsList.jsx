import PropTypes from 'prop-types';
import React from 'react';

import ScanningJobPanel from './ScanningJobPanel';
import scanningJobType from '../prop-types';

export default function ScanningJobsList(props) {
    const jobs = [];
    props.jobs.forEach((job) => {
        jobs.push(<ScanningJobPanel key={job.name} {...job} />);
    });
    return (
        <div>
            {jobs}
        </div>
    );
}

ScanningJobsList.propTypes = {
    jobs: PropTypes.arrayOf(scanningJobType).isRequired,
};
