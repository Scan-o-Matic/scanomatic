import PropTypes from 'prop-types';
import React from 'react';

export default class PolynomialResultsPlotScatter extends React.Component {

    constructor(props) {
        super(props);
        this.state = {
            d3Settings: {
                width: 600,
                height: 400,
                margins: {
                    left: 20,
                    right: 20,
                    top: 20,
                    bottom: 30,
                    axisLabel: 6,
                },
                markers: {
                    type: 'dot',
                    fill: 'rgba(120, 40, 40)',
                }
            }
        }
    }

    render() {
        return <div
            ref={ref => {this.divRef = ref;} }
            className='d3 d3-figure'
        />;
    }

    componentDidMount() {
        this.drawSVG2();
    }

    componentDidUpdate() {
        this.drawSVG2();
    }

    drawSVG2() {

        const yMax = Math.max(...this.props.resultsData.calculated);
        const xMax = Math.max(...this.props.resultsData.independentMeasurements);
        const rangeMax = Math.max(xMax, yMax) * 1.1;

        Plotly.newPlot(this.divRef, [{
            x: this.props.resultsData.independentMeasurements,
            y: this.props.resultsData.calculated,
            mode: 'markers',
            type: 'scatter',
            maker: {
                size: 12,
            }
        }], {
            title: 'Population Sizes',
            width: 500,
            height: 500,
            yaxis : {
                range: [0, rangeMax],
                title: 'Calculated',
            },
            xaxis : {
                range: [0, rangeMax],
                title: 'Independent Measurements',
            },
        });
    }
}

PolynomialResultsPlotScatter.propTypes = {
    resultsData: PropTypes.shape({
        calculated: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
    }).isRequired,
};
