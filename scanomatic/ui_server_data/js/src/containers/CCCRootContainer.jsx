import React from 'react';

import CCCRoot from '../components/CCCRoot';
import { InitiateCCC, finalizeCalibration } from '../api';


export default class CCCRootContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            cccMetadata: null,
            error: null,
            finalized: false,
        };
        this.handleError = this.handleError.bind(this);
        this.handleInitializeCCC = this.handleInitializeCCC.bind(this);
        this.handleFinalizeCCC = this.handleFinalizeCCC.bind(this);
    }

    handleInitializeCCC(species, reference, fixtureName, pinningFormat) {
        InitiateCCC(species, reference).then(
            ({ identifier: id, access_token: accessToken }) => this.setState({
                error: null,
                cccMetadata: {
                    id, accessToken, species, reference, fixtureName, pinningFormat,
                },
            }),
            reason => this.setState({
                error: `Error initializing calibration: ${reason}`,
            }),
        );
    }

    handleFinalizeCCC() {
        const { id, accessToken } = this.state.cccMetadata;
        finalizeCalibration(id, accessToken).then(
            () => this.setState({ error: null, finalized: true }),
            reason => this.setState({
                error: `Finalization error: ${reason}`,
            }),
        );
    }

    handleError(error) {
        this.setState({ error });
    }

    render() {
        return (
            <CCCRoot
                cccMetadata={this.state.cccMetadata}
                error={this.state.error}
                finalized={this.state.finalized}
                onInitializeCCC={this.handleInitializeCCC}
                onFinalizeCCC={this.handleFinalizeCCC}
                onError={this.handleError}
            />
        );
    }
}
