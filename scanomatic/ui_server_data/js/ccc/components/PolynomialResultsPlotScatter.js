import PropTypes from 'prop-types';
import React from 'react';

import c3 from 'c3';
import 'c3/c3.min.css';


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
        return <div className="poly-corr-chart"
            ref={ref => {this.divRef = ref;} }
        />;
    }

    componentDidMount() {
        this.drawSVG2();
    }

    componentDidUpdate() {
        this.drawSVG2();
    }

    drawSVG2() {

        const { calculated, independentMeasurements } = this.props.resultsData;
        const { slope, intercept, stderr } = this.props.correlation;
        const corr = x => (slope * x) + intercept;
        const yMax = Math.max(...calculated);
        const xMax = Math.max(...independentMeasurements);
        const rangeMax = Math.ceil(Math.max(xMax, yMax) * 1.1);
        c3.generate({
            bindto: this.divRef,
            data: {
                xs: {
                    calculated: 'independentMeasurements',
                    'Identity Line': 'identityLine_x',
                    Correlation: 'correlation_x',
                },
                columns: [
                    ['calculated'].concat(calculated),
                    ['independentMeasurements']
                        .concat(independentMeasurements),
                    ['Identity Line', 0, rangeMax],
                    ['identityLine_x', 0, rangeMax],
                    ['Correlation', corr(0), corr(rangeMax)],
                    ['correlation_x', 0, rangeMax],
                ],
                types: {
                    calculated: 'scatter',
                    'Identity Line': 'line',
                    Correlation: 'line',
                },
            },
            axis: {
                x: {
                    label: 'Independent Measurements',
                    tick: {
                        fit: false,
                        values: [0, rangeMax],
                    },
                },
                y: {
                    label: 'Calculated',
                    tick: {
                        fit: false,
                        values: [0, rangeMax],
                    },
                },
            },
            size: {
                width: 500,
                height: 500,
            },
            padding: {
                right: 6,
            },
            point: {
                r: 4,
            },
            legend: {
                hide: ['calculated'],
            },
        });
    }
}

PolynomialResultsPlotScatter.propTypes = {
    resultsData: PropTypes.shape({
        calculated: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
    }).isRequired,
    correlation: PropTypes.shape({
        slope: PropTypes.number.isRequired,
        intercept: PropTypes.number.isRequired,
        stderr: PropTypes.number.isRequired,
        p_value: PropTypes.number.isRequired,
    }).isRequired,
};
