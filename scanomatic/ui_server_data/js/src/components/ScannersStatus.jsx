import React from 'react';
import PropTypes from 'prop-types';
import Timeline from 'react-calendar-timeline/lib';
import moment from 'moment';
import 'react-calendar-timeline/lib/Timeline.css';

import '../../../style/status.css';

export function classNameForJob(job) {
    if (job.stopped) {
        return 'scanner-job scanner-job-stopped';
    } else if (job.end > new Date().getTime()) {
        return 'scanner-job scanner-job-running';
    }
    return 'scanner-job scanner-job-ended';
}

export default function ScannersStatus({ scanners, jobs }) {
    if (scanners.length === 0) {
        return (
            <div className="alert alert-danger" role="alert">
                <span className="glyphicon glyphicon-exclamation-sign" aria-hidden="true" />
                <span className="sr-only">Error:</span>
                No scanners attached to the system!
            </div>
        );
    }
    const groups = scanners
        .sort((a, b) => a.name > b.name)
        .map(s => ({
            id: s.id,
            title: s.name,
            rightTitle: s.isOnline ? <span className="label label-info">On</span> : <span className="label label-danger">Off</span>,
        }));

    const commonItemProps = {
        canMove: false,
        canResize: false,
        canChangeGroup: false,
    };

    const items = jobs
        .filter(j => j.started)
        .map(j => ({
            id: j.id,
            group: j.scannerId,
            title: j.name,
            start_time: j.started,
            end_time: j.stopped || j.end,
            className: classNameForJob(j),
            ...commonItemProps,
        }));
    return (
        <div className="timeline-group">
            <Timeline
                groups={groups}
                items={items}
                defaultTimeStart={moment().add(-100, 'hour')}
                defaultTimeEnd={moment().add(60, 'hour')}
                rightSidebarWidth={50}
                rightSidebarContent="Status"
                sidebarContent="Scanner"
            />
            <div className="row legend">
                <div className="col-md-6">
                    <div className="panel panel-default">
                        <div className="panel-heading">Legend</div>
                        <div className="panel-body">
                            <span className="legend-job-running">
                                Running Job
                            </span>
                            <span className="legend-job-stopped">
                                Stopped Job
                            </span>
                            <span className="legend-job-ended">
                                Ended Job
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

ScannersStatus.propTypes = {
    scanners: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string.isRequired,
        isOnline: PropTypes.bool.isRequired,
    })).isRequired,
    jobs: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        name: PropTypes.string.isRequired,
        scannerId: PropTypes.string.isRequired,
        started: PropTypes.number,
        stopped: PropTypes.number,
        end: PropTypes.number,
    })).isRequired,
};
