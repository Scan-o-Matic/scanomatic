import PropTypes from 'prop-types';
import React from 'react';

import PolynomialConstruction from '../components/PolynomialConstruction';

import * as API from '../api';
import CCCPropTypes from '../prop-types';


export default class PolynomialConstructionContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            degreeOfPolynomial: 5,
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
        this.handleDegreeOfPolynomialChange = this.handleDegreeOfPolynomialChange.bind(this);
    }

    handleConstruction() {
        const { id, accessToken } = this.props.cccMetadata;
        const { degreeOfPolynomial } = this.state;
        return API.SetNewCalibrationPolynomial(id, degreeOfPolynomial, accessToken)
            .then(this.handleConstructionResults)
            .catch(this.handleConstructionResultsError);
    }

    handleConstructionResults(results) {
        this.setState(
            {
                error: null,
                polynomial: {
                    power: results.polynomial_coefficients.length - 1,
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

    handleDegreeOfPolynomialChange({ target: { value } }) {
        this.setState({ degreeOfPolynomial: parseInt(value, 10) });
    }

    render() {
        return <PolynomialConstruction
            degreeOfPolynomial={this.state.degreeOfPolynomial}
            polynomial={this.state.polynomial}
            error={this.state.error}
            onClearError={this.handleClearError}
            onConstruction={this.handleConstruction}
            onDegreeOfPolynomialChange={this.handleDegreeOfPolynomialChange}
        />;
    }
}

PolynomialConstructionContainer.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata.isRequired,
};
