import PropTypes from 'prop-types';
import React from 'react';

import { createCanvasImage, getMarkerData } from '../helpers';
import CanvasState from '../CanvasState';
import Blob from '../Blob';

export default class ColonyImage extends React.Component {
    constructor(props) {
        super(props);
        this.handleClickPlus = this.handleClickPlus.bind(this);
        this.handleClickMinus = this.handleClickMinus.bind(this);
        this.handleClickUpdate = this.handleClickUpdate.bind(this);
    }

    componentDidMount() {
        this.updateCanvas();
    }

    componentDidUpdate() {
        this.updateCanvas();
    }

    updateCanvas() {
        createCanvasImage(this.props.data, this.colonyImageCanvas);
        if (this.props.draw) {
            const { width, height } = this.colonyImageCanvas;
            this.colonyMarkingsCanvas.width = width;
            this.colonyMarkingsCanvas.height = height;
            this.metaDataCanvasState = new CanvasState(this.colonyMarkingsCanvas);
            this.blob = new Blob(width / 2, height / 2, 15, "rgba(255, 0, 0, .2)");
            this.metaDataCanvasState.addShape(this.blob);
        }
    }

    handleClickPlus() {
        this.blob.r += 5;
        this.metaDataCanvasState.needsRender = true;
    }

    handleClickMinus() {
        this.blob.r -= 5;
        this.metaDataCanvasState.needsRender = true;
    }

    handleClickUpdate() {
        if (this.props.onUpdate) {
            const data = getMarkerData(this.colonyMarkingsCanvas);
            this.props.onUpdate(data);
        }
    }

    render() {
        const style = {
            position: 'relative',
            background: 'white',
            display: 'inline-block',
            padding: '3px',
        };

        const buttonStyle = { horizAlign: 'center' };
        return (
            <div>
                <div style={style}>
                    <canvas
                        ref={canvas => this.colonyImageCanvas = canvas}
                        style={{ zIndex: 1 }}
                    />
                    {this.props.draw &&
                        <canvas
                            ref={canvas => this.colonyMarkingsCanvas = canvas}
                            style={{ position: 'absolute', zIndex: 2, left:0, top: 0 }}
                        />}
                </div>
                {this.props.draw &&
                    <div>
                        <button
                            className="btn btn-default btn-plus"
                            style={buttonStyle}
                            onClick={this.handleClickPlus}
                        >+</button>
                        <button
                            className="btn btn-default btn-minus"
                            style={buttonStyle}
                            onClick={this.handleClickMinus}
                        >-</button>
                        <button
                            className="btn btn-default btn-update"
                            style={buttonStyle}
                            onClick={this.handleClickUpdate}
                        >Update</button>
                    </div>
                }
            </div>
        );
    }
}
ColonyImage.propTypes = {
    data: PropTypes.object.isRequired,
    onUpdate: PropTypes.func,
};
