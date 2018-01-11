import PropTypes from 'prop-types';
import React from 'react';

import { createCanvasMarker, featureColors } from '../helpers';

export default class ColonyFeatures extends React.Component {
    componentDidMount() {
        this.updateCanvas();
    }

    componentDidUpdate() {
        this.updateCanvas();
    }

    updateCanvas() {
        createCanvasMarker(this.props.data, this.canvas);
    }

    render() {
        const legend = [
            { color: featureColors.blob.toCSSString(), text: "Blob" },
            { color: featureColors.background.toCSSString(), text: "Background" },
            { color: featureColors.neither.toCSSString(), text: "Neither" },
        ];

        const style = {
            padding: '3px',
        };

        return (
            <div style={style} >
                <canvas ref={(canvas) => { this.canvas = canvas; }} />
                <ul className="colonyPlot">
                    {legend.map(({ color, text }) => (
                        <li key={text}>
                            <div
                                className="color-box"
                                style={{ backgroundColor: color }}
                            />
                            &nbsp;
                            {text}
                        </li>
                    ))}
                </ul>
            </div>
        );
    }
}
ColonyFeatures.propTypes = {
    data: PropTypes.object.isRequired,
};
