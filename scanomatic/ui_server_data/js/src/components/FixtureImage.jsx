import React from 'react';
import PropTypes from 'prop-types';


export default class FixtureImage extends React.Component {
    constructor(props) {
        super(props);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
    }

    componentDidMount() {
        this.ctx = this.canvas.getContext('2d');
        this.ctx.strokeStyle = '#F00';
        this.ctx.lineWidth = 3;
        this.addMouseEvents();
    }

    componentWillUnmount() {
        this.removeMouseEvents();
    }

    getMouseImagePosition(evt) {
        if (!this.canvas) return null;
        const { left, top } = this.canvas.getBoundingClientRect();
        return { x: evt.clientX - left, y: evt.clientY - top };
    }

    addMouseEvents() {
        document.addEventListener('mousemove', this.handleMouseMove, false);
        document.addEventListener('mouseup', this.handleMouseUp, false);
        document.addEventListener('mousedown', this.handleMouseDown, false);
    }

    removeMouseEvents() {
        document.removeEventListener('mousemove', this.handleMouseMove, false);
        document.remoteEventListener('mouseup', this.handleMouseUp, false);
        document.removeEventListener('mousedown', this.handleMouseDown, false);
    }

    handleMouseMove(evt) {
        const pos = this.getMouseImagePosition(evt);
        this.props.onMouse(pos);
    }

    handleMouseDown(evt) {
        const pos = this.getMouseImagePosition(evt);
        this.props.onAreaStart(pos);
    }

    handleMouseUp(evt) {
        const pos = this.getMouseImagePosition(evt);
        this.props.onAreaEnd(pos);
    }

    updateCanvas() {

    }

    render() {
        return (
            <canvas
                ref={(canvas) => {
                    this.canvas = canvas;
                }}
            />
        );
    }
}

FixtureImage.propTypes = {
    onAreaStart: PropTypes.func,
    onAreaEnd: PropTypes.func,
    onMouse: PropTypes.func,
};

FixtureImage.defaultProps = {
    onAreaStart: null,
    onAreaEnd: null,
    onMouse: null,
};
