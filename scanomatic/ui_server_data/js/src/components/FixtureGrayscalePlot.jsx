import React from 'react';
import PropTypes from 'prop-types';

import c3 from 'c3';
import 'c3/c3.min.css';

export default class FixtureGrayscalePlot extends React.PureComponent {
    componentDidMount() {
        this.drawSVG();
    }

    componentDidUpdate() {
        this.drawSVG();
    }

    drawSVG() {
        if (!this.ref) return;
        const {
            width, height, pixelValues, referenceValues,
        } = this.props;
        c3.generate({
            bindto: this.ref,
            data: {
                x: 'x',
                columns: [
                    ['x'].concat(pixelValues),
                    ['Mapping'].concat(referenceValues),
                ],
            },
            axis: {
                x: {
                    label: 'Pixel Values',
                    tick: {
                        fit: false,
                        values: [0, 63, 127, 191, 255],
                    },
                    min: 0,
                    max: 255,
                },
                y: {
                    label: 'Opacity',
                    tick: {
                        fit: false,
                        values: [0, 50, 100],
                    },
                    min: 0,
                    max: 100,
                },
            },
            size: {
                width,
                height,
            },
            legend: {
                show: false,
            },
        });
    }

    render() {
        const { pixelValues, referenceValues } = this.props;
        if (pixelValues == null || referenceValues == null) {
            this.ref = null;
            return <div className="alert alert-info fixture-grayscale">No grayscale detected</div>;
        }
        return <div className="fixture-grayscale" ref={(elem) => { this.ref = elem; }} />;
    }
}

FixtureGrayscalePlot.propTypes = {
    width: PropTypes.number,
    height: PropTypes.number,
    pixelValues: PropTypes.arrayOf(PropTypes.number),
    referenceValues: PropTypes.arrayOf(PropTypes.number),
};

FixtureGrayscalePlot.defaultProps = {
    width: null,
    height: null,
    pixelValues: null,
    referenceValues: null,
};
