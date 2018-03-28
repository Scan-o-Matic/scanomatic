import React from 'react';

import {
    deleteScanningJob,
    getScanningJobs,
    startScanningJob,
    terminateScanningJob,
} from '../api';
import { getScannersWithOwned } from '../helpers';
import ScanningRoot from '../components/ScanningRoot';
import { duration2milliseconds } from '../components/ScanningJobPanelBody';


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
        this.handleStartJob = this.handleStartJob.bind(this);
        this.handleStopJob = this.handleStopJob.bind(this);
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

    handleStartJob(startingJob) {
        const { jobs } = this.state;
        const newJobs = [];
        let foundJob = false;
        jobs.forEach((job) => {
            if (job.identifier === startingJob.identifier) {
                newJobs.push(Object.assign({}, job, { disableStart: true }));
                foundJob = true;
            } else {
                newJobs.push(job);
            }
        });
        if (foundJob) {
            this.setState({ jobs: newJobs });
        } else {
            this.setState({ error: `UI lost job '${startingJob.name}'` });
            return;
        }

        startScanningJob(startingJob)
            .then(() => {
                this.getJobsStatus();
            })
            .catch((reason) => {
                this.setState({ error: `Error starting job: ${reason}` });
            });
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

    handleStopJob(jobId) {
        this.setState({
            jobs: this.state.jobs.map(job =>
                (job.identifier === jobId
                    ? Object.assign({}, job, { status: 'Completed' })
                    : job)),
        });
        terminateScanningJob(jobId)
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
                onStartJob={this.handleStartJob}
                onRemoveJob={this.handleRemoveJob}
                onStopJob={this.handleStopJob}
            />
        );
    }
}
