import React from 'react';

import { getScanningJobs, startScanningJob } from '../api';
import { getScannersWithOwned } from '../helpers';
import ScanningRoot from '../components/ScanningRoot';
import { duration2milliseconds } from '../components/ScanningJobPanel';


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
        this.getJobStatusRequests(true);
    }

    getJobStatusRequests(monitor) {
        if (monitor) {
            setTimeout(() => this.getJobStatusRequests(true), 10000);
        }
        getScanningJobs()
            .then(jobs => this.setState({
                jobs: jobs.map(job => Object.assign({}, job, {
                    startTime: job.startTime ? new Date(job.startTime) : null,
                    endTime: job.startTime ?
                        new Date(new Date(job.startTime) - -duration2milliseconds(job.duration)) :
                        null,
                })),
            }))
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
        const newJobs = [];
        let foundJob = false;
        jobs.forEach((j) => {
            if (j.identifier === job.identifier) {
                newJobs.push(Object.assign({}, j, { disableStart: true }));
                foundJob = true;
            } else {
                newJobs.push(job);
            }
        });
        if (foundJob) {
            this.setState({ jobs: newJobs });
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
