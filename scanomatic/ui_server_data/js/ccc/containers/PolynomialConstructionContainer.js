import PropTypes from 'prop-types';
import React from 'react';

import PolynomialConstruction from '../components/PolynomialConstruction';

import * as API from '../api';
import CCCPropTypes from '../prop-types';


export default class PolynomialConstructionContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            power: 5,
            error: null,
            polynomial: null,
            resultsData: null,
        };

        this.handleConstruction = this.handleConstruction.bind(this);
        this.handleConstructionResults = this.handleConstructionResults
            .bind(this);
        this.handleConstructionResultsError =
            this.handleConstructionResultsError.bind(this);
        this.handleClearError = this.handleClearError.bind(this);
    }

    handleConstruction() {
        const { id, accessToken } = this.props.cccMetadata;
        const { power } = this.state;
        return API.SetNewCalibrationPolynomial(id, power, accessToken)
        .then(this.handleConstructionResults)
        .catch(this.handleConstructionResultsError);
    }

    handleConstructionResults(results) {
        this.setState(
            {
                error: null,
                polynomial: {
                    power: results.polynomial_power,
                    coefficients: results.polynomial_coefficients,
                    colonies: results.calculated_sizes.length,
                },
                resultsData: {
                    calculated: results.calculated_sizes,
                    independentMeasurements: results.measured_sizes
                },
            }
        );
    }

    handleConstructionResultsError(reason) {
        this.setState({ error: reason });
    }

    handleClearError() {
        this.setState({ error: null });
    }

    render() {
        return <PolynomialConstruction
            polynomial={this.state.polynomial}
            error={this.state.error}
            onClearError={this.handleClearError}
            onConstruction={this.handleConstruction}
        />;
    }
}

PolynomialConstructionContainer.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata.isRequired,
};
