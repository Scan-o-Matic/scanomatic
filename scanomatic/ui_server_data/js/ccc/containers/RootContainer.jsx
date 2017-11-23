import React from 'react';

import Root from '../components/Root';
import { InitiateCCC } from '../api';


export default class RootContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            cccMetadata: null,
            error: null,
        };
        this.handleError = this.handleError.bind(this);
        this.handleInitializeCCC = this.handleInitializeCCC.bind(this);
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

    handleError(error) {
        this.setState({ error });
    }

    render() {
        return (
            <Root
                cccMetadata={this.state.cccMetadata}
                error={this.state.error}
                onInitializeCCC={this.handleInitializeCCC}
                onError={this.handleError}
            />
        );
    }
}
