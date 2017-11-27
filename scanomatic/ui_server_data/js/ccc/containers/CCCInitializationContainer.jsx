import PropTypes from 'prop-types';
import React from 'react';

import CCCInitialization from '../components/CCCInitialization';
import { GetFixtures, GetPinningFormats } from '../api';


export default class CCCInitializationContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            species: '',
            reference: '',
            fixtureName: '',
            fixtureNames: [],
            pinningFormat: null,
            pinningFormats: [],
        };
        this.handleSpeciesChange = this.handleSpeciesChange.bind(this);
        this.handleReferenceChange = this.handleReferenceChange.bind(this);
        this.handleFixtureNameChange = this.handleFixtureNameChange.bind(this);
        this.handlePinningFormatNameChange = this.handlePinningFormatNameChange.bind(this);
        this.handleSumbit = this.handleSumbit.bind(this);
    }

    componentDidMount() {
        GetFixtures().then(
            this.handleGetFixturesSuccess.bind(this),
            this.setError.bind(this, 'Error getting fixtures'),
        );
        GetPinningFormats().then(
            this.handleGetPinningFormatsSuccess.bind(this),
            this.setError.bind(this, 'Error getting pinning formats'),
        );
    }

    setError(prefix, reason) {
        this.props.onError(`${prefix}: ${reason}`);
    }

    getPinningFormatByName(name) {
        for (let i = 0; i < this.state.pinningFormats.length; i += 1) {
            if (this.state.pinningFormats[i].name === name) {
                return this.state.pinningFormats[i];
            }
        }
        return null;
    }

    handleGetFixturesSuccess(fixtureNames) {
        if (fixtureNames.length > 0) {
            this.setState({ fixtureNames, fixtureName: fixtureNames[0] });
        } else {
            this.props.onError('You need to setup a fixture first.');
        }
    }

    handleGetPinningFormatsSuccess(pinningFormats) {
        this.setState({ pinningFormats, pinningFormat: pinningFormats[0] });
    }

    handleSpeciesChange(event) {
        this.setState({ species: event.target.value });
    }

    handleReferenceChange(event) {
        this.setState({ reference: event.target.value });
    }

    handleFixtureNameChange(event) {
        this.setState({ fixtureName: event.target.value });
    }

    handlePinningFormatNameChange(event) {
        const pinningFormat = this.getPinningFormatByName(event.target.value);
        this.setState({ pinningFormat });
    }

    handleSumbit() {
        if (!this.state.species) {
            this.props.onError('Species cannot be empty');
            return;
        }
        if (!this.state.reference) {
            this.props.onError('Reference cannot be empty');
            return;
        }
        if (!this.state.fixtureName) {
            this.props.onError('Fixture name cannot be empty');
            return;
        }
        if (!this.state.pinningFormat) {
            this.props.onError('Pinning format cannot be empty');
            return;
        }
        const {
            species, reference, fixtureName, pinningFormat,
        } = this.state;
        this.props.onInitialize(species, reference, fixtureName, pinningFormat);
    }

    render() {
        return (<CCCInitialization
            {...this.state}
            pinningFormatName={this.state.pinningFormat ? this.state.pinningFormat.name : ''}
            pinningFormatNames={this.state.pinningFormats.map(f => f.name)}
            onSpeciesChange={this.handleSpeciesChange}
            onReferenceChange={this.handleReferenceChange}
            onFixtureNameChange={this.handleFixtureNameChange}
            onPinningFormatNameChange={this.handlePinningFormatNameChange}
            onSubmit={this.handleSumbit}
        />);
    }
}

CCCInitializationContainer.propTypes = {
    onError: PropTypes.func.isRequired,
    onInitialize: PropTypes.func.isRequired,
};
