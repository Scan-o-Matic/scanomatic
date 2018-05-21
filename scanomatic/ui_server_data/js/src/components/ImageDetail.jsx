import React from 'react';
import PropTypes from 'prop-types';

import { loadImage } from '../helpers.js';

export default class ImageDetail extends React.Component {
    constructor(props) {
        super(props);
        this.state = { img: null };
    }

    componentDidMount() {
        if (!this.canvas) return;
        loadImage(this.props.imageUri)
            .then((img) => {
                this.setState({ img });
                this.props.onLoaded();
            });

        this.ctx = this.canvas.getContext('2d');
        this.ctx.lineWidth = 1;
        this.ctx.strokeStyle = '#0e0';
        this.ctx.translate(this.props.width, 0);
        this.ctx.scale(-1, 1);
        this.redrawDetail();
    }

    componentDidUpdate() {
        this.redrawDetail();
    }

    redrawDetail() {
        const { img } = this.state;
        const {
            x, y, width, height, crossHair,
        } = this.props;

        this.ctx.clearRect(0, 0, width, height);

        if (x == null || y == null) return;

        const centerViewX = Math.floor(width / 2) + 1;
        const centerViewY = Math.floor(height / 2) + 1;
        const outerGap = 0.25;
        const innerGap = 0.9;

        if (img) {
            this.ctx.drawImage(
                img,
                x - centerViewX,
                y - centerViewY,
                width,
                height,
                0,
                0,
                width,
                height,
            );
        }

        if (crossHair) {
            this.ctx.beginPath();
            this.ctx.moveTo(centerViewX * outerGap, centerViewY);
            this.ctx.lineTo(centerViewX * innerGap, centerViewY);
            this.ctx.moveTo(width - (centerViewX * outerGap), centerViewY);
            this.ctx.lineTo(width - (centerViewX * innerGap), centerViewY);
            this.ctx.moveTo(centerViewX, centerViewY * outerGap);
            this.ctx.lineTo(centerViewX, centerViewY * innerGap);
            this.ctx.moveTo(centerViewX, height - (centerViewY * outerGap));
            this.ctx.lineTo(centerViewX, height - (centerViewY * innerGap));
            this.ctx.stroke();
        }
    }

    render() {
        const { width, height } = this.props;
        return (
            <canvas
                className="image-detail"
                ref={(canvas) => { this.canvas = canvas; }}
                width={width}
                height={height}
            />
        );
    }
}

ImageDetail.propTypes = {
    crossHair: PropTypes.bool,
    x: PropTypes.number,
    y: PropTypes.number,
    width: PropTypes.number.isRequired,
    height: PropTypes.number.isRequired,
    imageUri: PropTypes.string.isRequired,
    onLoaded: PropTypes.func,
};

ImageDetail.defaultProps = {
    crossHair: false,
    onLoaded: () => {},
    x: null,
    y: null,
};
