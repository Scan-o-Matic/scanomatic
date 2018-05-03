import React from 'react';

import {
    deleteScanningJob,
    getScanningJobs,
} from '../api';
import { getScannersWithOwned } from '../helpers';
import ScanningRoot from '../components/ScanningRoot';


export default class ScanningRootContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            error: null,
            newJob: false,
            jobs: [],
            scanners: [],
        };
        this.handleCloseNewJob = this.handleCloseNewJob.bind(this);
        this.handleError = this.handleError.bind(this);
        this.handleNewJob = this.handleNewJob.bind(this);
        this.handleRemoveJob = this.handleRemoveJob.bind(this);
        this.handleUpdateFeed = this.getJobStatusRequests.bind(this);
    }

    componentDidMount() {
        this.getJobStatusRequests(true);
    }

    getJobStatusRequests(monitor) {
        if (monitor) {
            setTimeout(() => this.getJobStatusRequests(true), 10000);
        }
        getScanningJobs()
            .then(jobs => this.setState({
                jobs: jobs.map(job => Object.assign({}, job, {
                    endTime: job.startTime
                        ? job.duration.after(job.startTime)
                        : null,
                })),
            }))
            .catch(reason => this.setState({ error: `Error requesting jobs: ${reason}` }));

        getScannersWithOwned()
            .then(scanners => this.setState({ scanners }))
            .catch(reason => this.setState({ error: `Error requesting scanners: ${reason}` }));
    }

    handleCloseNewJob() {
        this.setState({ newJob: false, error: null });
        this.getJobStatusRequests();
    }

    handleError(error) {
        this.setState({ error });
    }

    handleNewJob() {
        this.setState({ newJob: true });
    }

    handleRemoveJob(jobId) {
        this.setState({ jobs: this.state.jobs.filter(job => job.identifier !== jobId) });
        deleteScanningJob(jobId)
            .catch((message) => {
                this.setState({ error: `Error deleting job: ${message}` });
            })
            .then(() => {
                this.getJobStatusRequests();
            });
    }

    render() {
        return (
            <ScanningRoot
                newJob={this.state.newJob}
                error={this.state.error}
                jobs={this.state.jobs}
                scanners={this.state.scanners}
                onError={this.handleError}
                onCloseNewJob={this.handleCloseNewJob}
                onNewJob={this.handleNewJob}
                onRemoveJob={this.handleRemoveJob}
                updateFeed={this.handleUpdateFeed}
            />
        );
    }
}
