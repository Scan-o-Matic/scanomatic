import PropTypes from 'prop-types';
import React from 'react';

import c3 from 'c3';
import 'c3/c3.min.css';

import { valueFormatter } from '../helpers.js';

const labelFormatter = value => value.toFixed(0);


export default class PolynomialResultsColonyHistogram extends React.Component {
    constructor(props) {
        super(props);
        this.refBinder = this.refBinder.bind(this);
    }

    componentDidMount() {
        this.drawSVG();
    }

    componentDidUpdate() {
        this.drawSVG();
    }

    drawSVG() {
        const {
            pixelValues, pixelCounts, maxCount, minPixelValue, maxPixelValue,
            colonyIdx,
        } = this.props;

        const xTicks = [
            minPixelValue,
            ((maxPixelValue - minPixelValue) / 2) + minPixelValue,
            maxPixelValue,
        ];
        const yTicks = [0, maxCount / 2, maxCount];

        c3.generate({
            bindto: this.divRef,
            data: {
                xs: {
                    Counts: 'Pixel Values',
                },
                columns: [
                    ['Counts'].concat(pixelCounts),
                    ['Pixel Values']
                        .concat(pixelValues),
                ],
                types: {
                    Counts: 'area-step',
                },
            },
            axis: {
                x: {
                    label: colonyIdx === 0 ? 'Pixel Values' : '',
                    tick: {
                        fit: false,
                        values: xTicks,
                        format: labelFormatter,
                        outer: false,
                    },
                    padding: { left: 0, right: 0 },
                    min: minPixelValue,
                    max: maxPixelValue,
                },
                y: {
                    label: colonyIdx === 0 ? 'Counts' : '',
                    tick: {
                        fit: false,
                        values: yTicks,
                        format: labelFormatter,
                        outer: false,
                    },
                    padding: { top: 0, bottom: 0 },
                    min: 0,
                    max: maxCount,
                },
            },
            size: {
                width: 520,
                height: 100,
            },
            padding: {
                right: 12,
                top: 8,
                bottom: 0,
            },
            legend: {
                show: false,
            },
        });
    }

    refBinder(ref) {
        this.divRef = ref;
    }

    render() {
        const { independentMeasurement, colonyIdx } = this.props;

        return (
            <div className="poly-colony-container">
                <div
                    className="poly-colony-chart"
                    ref={this.refBinder}
                    id={`poly-colony-chart-${colonyIdx}`}
                />
                <span className="poly-colony-txt">{valueFormatter(independentMeasurement, 2)} cells</span>
            </div>
        );
    }
}

PolynomialResultsColonyHistogram.propTypes = {
    colonyIdx: PropTypes.number.isRequired,
    pixelValues: PropTypes.arrayOf(PropTypes.number).isRequired,
    pixelCounts: PropTypes.arrayOf(PropTypes.number).isRequired,
    independentMeasurement: PropTypes.number.isRequired,
    maxCount: PropTypes.number.isRequired,
    maxPixelValue: PropTypes.number.isRequired,
    minPixelValue: PropTypes.number.isRequired,
};
