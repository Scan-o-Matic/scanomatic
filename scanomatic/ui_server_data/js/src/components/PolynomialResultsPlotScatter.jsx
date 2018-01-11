import PropTypes from 'prop-types';
import React from 'react';

import c3 from 'c3';
import 'c3/c3.min.css';
import { valueFormatter } from '../helpers.js';


export default class PolynomialResultsPlotScatter extends React.Component {
    componentDidMount() {
        this.drawSVG();
    }

    componentDidUpdate() {
        this.drawSVG();
    }

    drawSVG() {
        const { calculated, independentMeasurements } = this.props.resultsData;
        const { slope, intercept } = this.props.correlation;
        const corr = x => (slope * x) + intercept;
        const yMax = Math.max(...calculated);
        const xMax = Math.max(...independentMeasurements);
        const rangeMax = Math.ceil(Math.max(xMax, yMax) * 1.1);
        return c3.generate({
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
                        format: valueFormatter,
                    },
                    padding: { left: 0, right: 0 },
                },
                y: {
                    label: 'Calculated',
                    tick: {
                        fit: false,
                        values: [0, rangeMax],
                        format: valueFormatter,
                    },
                    padding: { top: 0, bottom: 0 },
                },
            },
            size: {
                width: 520,
                height: 500,
            },
            padding: {
                right: 30,
                top: 10,
            },
            point: {
                r: 4,
            },
            legend: {
                hide: ['calculated'],
            },
            tooltip: {
                format: {
                    title: d => `Measured ${valueFormatter(d, 2)}`,
                    value: value => valueFormatter(value, 2),
                    name: () => 'Calculated',
                },
            },
        });
    }

    render() {
        const { slope, intercept, stderr } = this.props.correlation;
        return (
            <div className="poly-corr">
                <h4>Population Size Correlation</h4>
                <div
                    className="poly-corr-chart"
                    ref={(ref) => { this.divRef = ref; }}
                />
                <p>Correlation:
                    y = {slope.toFixed(2)}x + {intercept.toFixed(0)}
                    {' '}
                    (standard error {stderr.toFixed(2)})
                </p>
            </div>
        );
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
    }).isRequired,
};
