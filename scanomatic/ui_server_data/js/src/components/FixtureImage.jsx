import React from 'react';
import PropTypes from 'prop-types';

const MIN_SELECTION = 10;

function getRect(pos1, pos2) {
    return {
        x: Math.min(pos1.x, pos2.x),
        y: Math.min(pos1.y, pos2.y),
        w: Math.abs(pos1.x - pos2.x) + 1,
        h: Math.abs(pos1.y - pos2.y) + 1,
    };
}

function getCenter(rect) {
    return {
        x: rect.x + (0.5 * rect.w),
        y: rect.y + (0.5 * rect.h),
    };
}

function getFontSize({ w, h }) {
    return Math.round(Math.max(10, (Math.min(Math.abs(w), Math.abs(h)) * 0.7)));
}

export default class FixtureImage extends React.Component {
    constructor(props) {
        super(props);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.state = { editMode: props.onAreaStart && props.onAreaEnd };
        this.overlayCanvas = null;
        this.isDragging = false;
        this.startPos = null;
        this.currentPos = null;
        this.scale = 10;
    }

    componentDidMount() {
        this.ctx = this.overlayCanvas.getContext('2d');
        this.ctx.strokeStyle = '#0e0';
        this.ctx.lineWidth = 2;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.addMouseEvents();
        this.updateCanvas();
        this.bgCtx = this.imageCanvas.getContext('2d');
        this.props.image.onload = () => this.drawBackground();
        this.drawBackground();
    }

    componentDidUpdate() {
        this.updateCanvas();
    }

    componentWillUnmount() {
        this.removeMouseEvents();
    }

    getMouseImagePosition(evt) {
        if (!this.overlayCanvas) return null;
        const { left, top } = this.overlayCanvas.getBoundingClientRect();
        let x = Math.max(0, this.props.width - (evt.clientX - left));
        x = Math.min(x, this.props.width - 1) * this.scale;
        let y = Math.max(0, evt.clientY - top);
        y = Math.min(y, this.props.height - 1) * this.scale;
        return { x, y };
    }

    addMouseEvents() {
        document.addEventListener('mousemove', this.handleMouseMove, false);
        document.addEventListener('mouseup', this.handleMouseUp, false);
        document.addEventListener('mousedown', this.handleMouseDown, false);
    }

    removeMouseEvents() {
        document.removeEventListener('mousemove', this.handleMouseMove, false);
        document.removeEventListener('mouseup', this.handleMouseUp, false);
        document.removeEventListener('mousedown', this.handleMouseDown, false);
    }

    handleMouseMove(evt) {
        const pos = this.getMouseImagePosition(evt);
        if (this.props.onMouse) this.props.onMouse(pos);
        if (!this.isDragging) return;
        this.currentPos = pos;
        this.updateCanvas();
    }

    handleMouseDown(evt) {
        if (!this.state.editMode) return;
        const pos = this.getMouseImagePosition(evt);
        this.isDragging = true;
        this.startPos = pos;
        this.currentPos = pos;
        if (this.props.onAreaStart) this.props.onAreaStart(pos);
        this.updateCanvas();
    }

    handleMouseUp(evt) {
        if (!this.state.editMode) return;
        const pos = this.getMouseImagePosition(evt);
        if (!this.startPos && this.props.onAreaEnd) this.props.onAreaEnd(null);
        const rect = getRect(this.startPos, pos);
        this.isDragging = false;
        this.startPos = null;
        this.currentPos = null;
        if (this.props.onAreaEnd) {
            this.props.onAreaEnd(rect.w > MIN_SELECTION && rect.h > MIN_SELECTION ? pos : null);
        }
        this.updateCanvas();
    }

    imagePosToCanvasPos({ x, y }) {
        return { x: ((this.props.width * this.scale) - x) / this.scale, y: y / this.scale };
    }

    imageRectToCanvasRect({
        x, y, w, h,
    }) {
        return {
            x: ((this.props.width * this.scale) - x) / this.scale,
            y: y / this.scale,
            w: -w / this.scale,
            h: h / this.scale,
        };
    }

    drawBackground() {
        const { width, height, image } = this.props;
        this.bgCtx.clearRect(0, 0, width, height);
        this.bgCtx.drawImage(image, 0, 0, width, height);
        this.bgCtx.translate(width, 0);
        this.bgCtx.scale(-1, 1);
    }

    updateCanvas() {
        // All positions are in image coordinates.
        const {
            areas, markers, width, height,
        } = this.props;

        this.ctx.clearRect(0, 0, width, height);

        // Paint markers
        const radius = 70 / this.scale;
        markers.forEach((m) => {
            const center = this.imagePosToCanvasPos(m);
            this.ctx.beginPath();
            this.ctx.arc(center.x, center.y, 2 * radius, 0, 2 * Math.PI, false);
            this.ctx.stroke();

            this.ctx.beginPath();
            this.ctx.moveTo(center.x - radius, center.y - radius);
            this.ctx.lineTo(center.x + radius, center.y + radius);
            this.ctx.moveTo(center.x + radius, center.y - radius);
            this.ctx.lineTo(center.x - radius, center.y + radius);
            this.ctx.stroke();
        });

        // Paint areas
        areas.forEach((a) => {
            const rect = this.imageRectToCanvasRect(a.rect);
            this.ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
            this.ctx.font = `${getFontSize(rect)} pt Calibri`;
            const center = getCenter(rect);
            this.ctx.fillText(a.name, center.x, center.y);
        });

        // Paint area being made
        if (this.isDragging) {
            const rect = getRect(this.startPos, this.currentPos);
            const cRect = this.imageRectToCanvasRect(rect);
            this.ctx.strokeRect(cRect.x, cRect.y, cRect.w, cRect.h);
            this.ctx.font = `${getFontSize(cRect)}pt Calibri`;
            const center = getCenter(cRect);
            this.ctx.fillStyle = '#0e0';
            this.ctx.fillText('?', center.x, center.y);
        }
    }

    render() {
        return (
            <div>
                <canvas
                    width={this.props.width}
                    height={this.props.height}
                    ref={(canvas) => {
                        this.imageCanvas = canvas;
                    }}
                    style={{
                        position: 'absolute',
                        zIndex: 1,
                        left: 0,
                        top: 0,
                    }}
                />
                <canvas
                    width={this.props.width}
                    height={this.props.height}
                    ref={(canvas) => {
                        this.overlayCanvas = canvas;
                    }}
                    style={{
                        position: 'absolute',
                        zIndex: 2,
                        left: 0,
                        top: 0,
                    }}
                />
            </div>
        );
    }
}

FixtureImage.propTypes = {
    image: PropTypes.instanceOf(Image).isRequired,
    markers: PropTypes.arrayOf(PropTypes.shape({
        x: PropTypes.number.isRequired,
        y: PropTypes.number.isRequired,
    })).isRequired,
    areas: PropTypes.arrayOf(PropTypes.shape({
        name: PropTypes.string.isRequired,
        rect: PropTypes.shape({
            x: PropTypes.number.isRequired,
            y: PropTypes.number.isRequired,
            w: PropTypes.number.isRequired,
            h: PropTypes.number.isRequired,
        }).isRequired,
    })),
    height: PropTypes.number,
    width: PropTypes.number,
    onAreaStart: PropTypes.func,
    onAreaEnd: PropTypes.func,
    onMouse: PropTypes.func,
};

FixtureImage.defaultProps = {
    width: 480,
    height: 600,
    onAreaStart: null,
    onAreaEnd: null,
    onMouse: null,
};
