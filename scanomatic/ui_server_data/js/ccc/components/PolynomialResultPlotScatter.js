import PropTypes from 'prop-types';
import React from 'react';

export default class PolynomialResultsPlotScatter extends React.Component {

    constructor(props) {
        super(props);
        this.divRef = null;
        this.d3Settings = {
            width: '80%',

        }
    }

    render() {
        return <div ref={(ref) => {this.divRef = ref;} } className='d3 d3-figure'></div>
    }

    componentDidUpdate() {
        if (this.divRef) {
            const svg ''
        }
    }
}

PolynomialResultsPlotScatter.propTypes = {
    resultsData: PropTypes.shape({
        calculated: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
    }),
};
