import React from 'react';

import { getJobs } from '../api';
import Root from '../components/Root';


export default class RootContainer extends React.Component {
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

    handleNewJob() {
        this.setState({ newJob: true });
    }

    handleCloseNewJob() {
        this.getJobs();
        this.setState({ newJob: false });
    }

    handleError(error) {
        this.setState({ error });
    }

    getJobs() {
        getJobs()
            .then(jobs => this.setState({ jobs, error: null }))
            .catch(reason => this.setState({ error: `Error requesting jobs: ${reason}` }));
    }

    render() {
        return (
            <Root
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
