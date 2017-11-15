import PropTypes from 'prop-types';
import React from 'react';

import Polynomial from '../components/Polynomial';

import { SetNewCalibrationPolynomial } from '../api';


export default class ResultsContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            power: 5,
            error: null,
            polynomial: null,
            resultsData: null,
        };

        this.handleConstruction = this.handleConstruction.bind(this);
        this.handleConstructionResults = this.handleConstructionResults.bind(this);
        this.handleConstructionResultsError = this.handleConstructionResultsError.bind(this);
        this.handleClearError = this.handleClearError.bind(this);
    }

    handleConstruction() {
        const { cccId, accessToken } = this.props;
        const { power } = this.state;
        SetNewCalibrationPolynomial(cccId, power, accessToken)
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
                },
                resultsData: {
                    calculated: results.calculated_sizes,
                    independentMeasurements: results.measured_sizes
                }

            }
        );
    }

    handleConstructionResultsError(error) {
        this.setState(
            {
                error: error.reason,
            }
        );
    }

    handleClearError() {
        this.setState(
            {
                error: null,
            }
        );
    }

    render() {
        return <Polynomial
            power={this.state.power}
            polynomial={this.state.polynomial}
            data={this.state.resultsData}
            error={this.state.error}
            handleClearError={this.handleClearError}
            handleConstruction={this.handleConstruction}
        />;
    }
}

ResultsContainer.propTypes = {
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
};
