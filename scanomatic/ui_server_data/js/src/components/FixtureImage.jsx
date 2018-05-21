import React from 'react';
import PropTypes from 'prop-types';

import { loadImage } from '../helpers.js';

const MIN_SELECTION = 10;
const MARKER_RADIUS_600DPI = 70;
const MIN_FONT_SIZE = 14;

export function getRect(pos1, pos2) {
    return {
        x: Math.min(pos1.x, pos2.x),
        y: Math.min(pos1.y, pos2.y),
        w: Math.abs(pos1.x - pos2.x) + 1,
        h: Math.abs(pos1.y - pos2.y) + 1,
    };
}

export function getCenter(rect) {
    return {
        x: rect.x + (0.5 * rect.w),
        y: rect.y + (0.5 * rect.h),
    };
}

export function getFontSize({ w, h }) {
    return Math.round(Math.max(MIN_FONT_SIZE, (Math.min(Math.abs(w), Math.abs(h)) * 0.7)));
}

export default class FixtureImage extends React.Component {
    constructor(props) {
        super(props);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.state = { editMode: props.onAreaStart && props.onAreaEnd, loaded: false };
        this.imageCanvas = null;
        this.overlayCanvas = null;
        this.isDragging = false;
        this.startPos = null;
        this.currentPos = null;
        this.scale = 10;
    }

    componentDidMount() {
        loadImage(this.props.imageUri)
            .then((img) => {
                this.setState({
                    loaded: true,
                    width: img.width / this.scale,
                    height: img.height / this.scale,
                    img,
                });
            })
            .catch(() => {
                this.setState({ error: 'Could not load fixture image!' });
                if (this.props.onLoaded) this.props.onLoaded();
            });
    }

    componentWillReceiveProps(nextProps) {
        if (nextProps.imageUri && nextProps.imageUri !== this.props.imageUri) {
            this.setState({
                loaded: false,
            });
            loadImage(nextProps.imageUri)
                .then((img) => {
                    const width = img.width / this.scale;
                    const height = img.height / this.scale;
                    if (this.state.width !== width || this.state.height !== height) {
                        this.ctx = null;
                        this.bgCtx = null;
                    }
                    this.setState({
                        loaded: true,
                        img,
                        width,
                        height,
                    });
                })
                .catch(() => {
                    this.setState({ error: 'Could not load fixture image!' });
                    if (this.props.onLoaded) this.props.onLoaded();
                });
        }
    }

    componentDidUpdate() {
        if (!this.overlayCanvas) return;
        if (!this.ctx || !this.bgCtx) this.initCanvases();
        this.updateCanvas();
    }

    componentWillUnmount() {
        this.removeMouseEvents();
    }

    getMouseImagePosition(evt) {
        if (!this.overlayCanvas) return null;
        const { left, top } = this.overlayCanvas.getBoundingClientRect();
        let x = Math.max(0, this.state.width - (evt.clientX - left));
        x = Math.min(x, this.state.width - 1) * this.scale;
        let y = Math.max(0, evt.clientY - top);
        y = Math.min(y, this.state.height - 1) * this.scale;
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
        if (rect.w < MIN_SELECTION && rect.h < MIN_SELECTION && this.props.onClick) {
            this.props.onClick(pos);
        }
        this.updateCanvas();
    }

    initCanvases() {
        this.ctx = this.overlayCanvas.getContext('2d');
        this.ctx.strokeStyle = '#0e0';
        this.ctx.lineWidth = 2;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillStyle = '#fff';
        this.ctx.shadowColor = '#000';
        this.ctx.shadowBlur = 8;

        this.bgCtx = this.imageCanvas.getContext('2d');
        this.bgCtx.translate(this.state.width, 0);
        this.bgCtx.scale(-1, 1);

        this.drawBackground();
        this.updateCanvas();
        this.addMouseEvents();
        if (this.props.onLoaded) this.props.onLoaded();
    }

    imagePosToCanvasPos({ x, y }) {
        return { x: ((this.state.width * this.scale) - x) / this.scale, y: y / this.scale };
    }

    imageRectToCanvasRect({
        x, y, w, h,
    }) {
        return {
            x: ((this.state.width * this.scale) - x) / this.scale,
            y: y / this.scale,
            w: -w / this.scale,
            h: h / this.scale,
        };
    }

    drawBackground() {
        const { width, height, img } = this.state;
        this.bgCtx.drawImage(img, 0, 0, width, height);
    }

    updateCanvas() {
        // All positions are in image coordinates.
        const { areas, markers } = this.props;
        const { width, height } = this.state;

        this.ctx.clearRect(0, 0, width, height);

        // Paint markers
        const radius = MARKER_RADIUS_600DPI / this.scale;
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
            this.ctx.font = `${getFontSize(rect)}pt Calibri`;
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
            this.ctx.fillText('?', center.x, center.y);
        }
    }

    render() {
        if (this.state.error) {
            return (
                <div className="alert alert-danger">
                    {this.state.error}
                </div>
            );
        } else if (!this.state.loaded) {
            return (
                <div className="alert alert-info">
                    Loading...
                </div>
            );
        }

        return (
            <div>
                <canvas
                    width={this.state.width}
                    height={this.state.height}
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
                    width={this.state.width}
                    height={this.state.height}
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
    imageUri: PropTypes.string.isRequired,
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
    })).isRequired,
    onAreaStart: PropTypes.func,
    onAreaEnd: PropTypes.func,
    onMouse: PropTypes.func,
    onClick: PropTypes.func,
    onLoaded: PropTypes.func,
};

FixtureImage.defaultProps = {
    onAreaStart: null,
    onAreaEnd: null,
    onMouse: null,
    onClick: null,
    onLoaded: null,
};
