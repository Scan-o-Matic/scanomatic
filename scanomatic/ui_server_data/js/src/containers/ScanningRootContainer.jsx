import React from 'react';

import { getScanningJobs, startScanningJob } from '../api';
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
        this.handleError = this.handleError.bind(this);
        this.handleNewJob = this.handleNewJob.bind(this);
        this.handleCloseNewJob = this.handleCloseNewJob.bind(this);
        this.handleStartJob = this.handleStartJob.bind(this);
    }

    componentDidMount() {
        this.getJobStatusRequests();
    }

    getJobStatusRequests() {
        getScanningJobs()
            .then(jobs => this.setState({ jobs }))
            .catch(reason => this.setState({ error: `Error requesting jobs: ${reason}` }));

        getScannersWithOwned()
            .then(scanners => this.setState({ scanners }))
            .catch(reason => this.setState({ error: `Error requesting scanners: ${reason}` }));
    }

    getJobsStatus() {
        this.setState({ error: null });
        this.getJobStatusRequests();
    }

    handleCloseNewJob() {
        this.getJobsStatus();
        this.setState({ newJob: false });
    }

    handleError(error) {
        this.setState({ error });
    }

    handleNewJob() {
        this.setState({ newJob: true });
    }

    handleStartJob(job) {
        const { jobs } = this.state;
        const jobInQuestion = jobs.filter(j => j.identifier === job.identifier);
        if (jobInQuestion.length === 1) {
            jobInQuestion.disableStart = true;
            this.setState({ jobs });
        } else {
            this.setState({ error: `UI lost job '${job.name}'` });
            return;
        }

        startScanningJob(job)
            .then(() => {
                this.getJobsStatus();
            })
            .catch((reason) => {
                this.setState({ error: `Error starting job: ${reason}` });
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
                onStartJob={this.handleStartJob}
            />
        );
    }
}
