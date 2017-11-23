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

        const yMax = Math.max(...this.props.resultsData.calculated);
        const xMax = Math
            .max(...this.props.resultsData.independentMeasurements);
        const rangeMax = Math.max(xMax, yMax) * 1.1;
        const { calculated, independentMeasurements } = this.props.resultsData;
        const plot = c3.generate({
            bindto: this.divRef,
            data: {
                xs: {
                    calculated: 'independentMeasurements',
                    identityLine: 'identityLine_x',
                },
                columns: [
                    ['calculated'].concat(calculated),
                    ['independentMeasurements']
                        .concat(independentMeasurements),
                    ['identityLine', 0, rangeMax],
                    ['identityLine_x', 0, rangeMax],
                ],
                types: {
                    calculated: 'scatter',
                    identityLine: 'line',
                }
            },
            axis: {
                x: {
                    label: 'Independent Measurements',
                    min: 0,
                    max: rangeMax,
                    extent: [0, rangeMax],
                },
                y: {
                    label: 'Calculated',
                    min: 0,
                    max: rangeMax,
                    extent: [0, rangeMax],
                }
            },
            size: {
                width: 500,
                height: 500,
            },
            point: {
                r: 4,
            },
            legend: {
                show: false,
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
