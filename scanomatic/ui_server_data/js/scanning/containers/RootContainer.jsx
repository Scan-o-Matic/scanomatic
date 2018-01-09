import React from 'react';

import { submitJob } from '../api';
import Root from '../components/Root';


export default class RootContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            error: null,
            newJob: false,
        };
        this.handleError = this.handleError.bind(this);
        this.handleNewJob = this.handleNewJob.bind(this);
        this.handleCloseNewJob = this.handleCloseNewJob.bind(this);
    }

    handleNewJob() {
        this.setState({ newJob: true });
    }

    handleCloseNewJob() {
        this.setState({ newJob: false });
    }

    handleError(error) {
        this.setState({ error });
    }

    render() {
        return (
            <Root
                newJob={this.state.newJob}
                error={this.state.error}
                onError={this.handleError}
                onCloseNewJob={this.handleCloseNewJob}
                onNewJob={this.handleNewJob}
            />
        );
    }
}
