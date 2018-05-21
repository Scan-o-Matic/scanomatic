import React from 'react';
import PropTypes from 'prop-types';

import ScannersStatus from './ScannersStatus';

export default function StatusRoot({ hasLoaded, scanners, experiments }) {
    let content;
    if (hasLoaded) {
        content = <ScannersStatus scanners={scanners} jobs={experiments} />;
    } else {
        content = <div className="alert alert-info">Loading...</div>;
    }
    return (
        <div>
            <h2>Scanners</h2>
            {content}
        </div>
    );
}

StatusRoot.propTypes = {
    hasLoaded: PropTypes.bool.isRequired,
    scanners: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string.isRequired,
        isOnline: PropTypes.bool.isRequired,
    })).isRequired,
    experiments: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string.isRequired,
        scannerId: PropTypes.string.isRequired,
        started: PropTypes.number,
        stopped: PropTypes.number,
        end: PropTypes.number,
    })).isRequired,
};
