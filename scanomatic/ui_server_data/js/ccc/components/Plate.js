import PropTypes from 'prop-types';
import React from 'react';

const COLONY_OUTLINE_RADIUS = 30;
const SELECTED_COLONY_MARKER_RADIUS = 40;
const SCALE = .2;
const FIRST_COLONY_FILL_COLOR = "#c82124";

export function drawCircle(context, x, y, radius) {
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
    context.stroke();
}

export function drawDisk(context, x, y, radius, color) {
    context.beginPath();
    context.arc(x, y, radius, 0, 2 * Math.PI);
    context.closePath();
    context.fillStyle = color;
    context.fill();
}

export default class Plate extends React.Component {
    render() {
        return <canvas
            width={this.props.image.width * SCALE}
            height={this.props.image.height * SCALE}
            ref={canvas => this.canvas = canvas}
        />;
    }

    componentDidMount() {
        this.updateCanvas();
    }

    componentDidUpdate() {
        this.updateCanvas();
    }

    updateCanvas() {
        const context = this.canvas.getContext('2d');
        const { width, height } = this.canvas;
        context.drawImage(this.props.image, 0, 0, width, height);
        context.scale(SCALE, SCALE);
        if (this.props.grid) {
            const [ys, xs] = this.props.grid;
            for(let row = 0; row < xs.length; row++) {
                for (let col = 0; col < xs[row].length; col++) {
                    const x = xs[row][col];
                    const y = ys[row][col];
                    if (row == 0 && col == 0) {
                        drawDisk(context, x, y, COLONY_OUTLINE_RADIUS, FIRST_COLONY_FILL_COLOR)
                    } else {
                        drawCircle(context, x, y, COLONY_OUTLINE_RADIUS);
                    }
                }
            }

            if (this.props.selectedColony) {
                const { row, col } = this.props.selectedColony;
                const x = xs[row][col];
                const y = ys[row][col];
                drawCircle(context, x, y, SELECTED_COLONY_MARKER_RADIUS);
            }
        }
        context.scale(1/SCALE, 1/SCALE);
    }
}

Plate.propTypes = {
    image: PropTypes.instanceOf(Image).isRequired,
    grid: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number))),
    selectedColony: PropTypes.shape({ row: PropTypes.number, col: PropTypes.number }),
};
