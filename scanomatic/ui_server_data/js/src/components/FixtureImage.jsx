import React from 'react';
import PropTypes from 'prop-types';


export default class FixtureImage extends React.Component {
    constructor(props) {
        super(props);
        this.handleMouseMove = this.handleMouseMove.bind(this);
        this.handleMouseUp = this.handleMouseUp.bind(this);
        this.handleMouseDown = this.handleMouseDown.bind(this);
        this.setState({ editMode: props.onAreaStart && props.onAreaEnd });
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
        let x = Math.max(0, this.props.width - (evt.clientX - left));
        x = Math.min(x, this.props.width - 1);
        let y = Math.max(0, evt.clientY - top);
        y = Math.min(y, this.props.height - 1);
        return { x, y };
    }

    canvas = null;
    isDragging = false;
    startPos = null;
    currentPos = null;

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
        if (!this.isDragging) return;
        const pos = this.getMouseImagePosition(evt);
        this.currentPos = pos;
        if (this.props.onMouse) this.props.onMouse(pos);
    }

    handleMouseDown(evt) {
        if (!this.state.editMode) return;
        const pos = this.getMouseImagePosition(evt);
        this.isDragging = true;
        this.startPos = pos;
        this.currentPos = pos;
        if (this.props.onAreaStart) this.props.onAreaStart(pos);
    }

    handleMouseUp(evt) {
        if (!this.state.editMode) return;
        const pos = this.getMouseImagePosition(evt);
        this.isDragging = false;
        this.startPos = null;
        this.currentPos = null;
        if (this.props.onAreaEnd) this.props.onAreaEnd(pos);
    }

    updateCanvas() {

    }

    render() {
        return (
            <canvas
                width={this.props.width}
                height={this.props.height}
                ref={(canvas) => {
                    this.canvas = canvas;
                }}
            />
        );
    }
}

FixtureImage.propTypes = {
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
