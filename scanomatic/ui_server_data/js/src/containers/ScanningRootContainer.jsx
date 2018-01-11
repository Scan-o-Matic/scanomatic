import React from 'react';

import { getScanningJobs } from '../api';
import ScanningRoot from '../components/ScanningRoot';


export default class ScanningRootContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            error: null,
            newJob: false,
            jobs: [],
        };
        this.handleError = this.handleError.bind(this);
        this.handleNewJob = this.handleNewJob.bind(this);
        this.handleCloseNewJob = this.handleCloseNewJob.bind(this);
        this.getJobs = this.getJobs.bind(this);
    }

    componentDidMount() {
        this.getJobs();
    }

    getJobs() {
        getScanningJobs()
            .then(jobs => this.setState({ jobs, error: null }))
            .catch(reason => this.setState({ error: `Error requesting jobs: ${reason}` }));
    }

    handleCloseNewJob() {
        this.getJobs();
        this.setState({ newJob: false });
    }

    handleError(error) {
        this.setState({ error });
    }

    handleNewJob() {
        this.setState({ newJob: true });
    }

    render() {
        return (
            <ScanningRoot
                newJob={this.state.newJob}
                error={this.state.error}
                jobs={this.state.jobs}
                onError={this.handleError}
                onCloseNewJob={this.handleCloseNewJob}
                onNewJob={this.handleNewJob}
            />
        );
    }
}
