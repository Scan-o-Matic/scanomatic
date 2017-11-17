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
            background: 'white',
            display: 'inline-block',
            paddingd: '3px',
            textAlign: 'center',
        };

        return (
            <div style={style} >
                <canvas ref={canvas => this.canvas = canvas} />;
                <ul className="colonyPlot" style={ { color: 'black' } }>
                    {legend.map( ({ color, text }) => {
                        const style = { backgroundColor: color };
                        return (
                            <li key={text}>
                                <div className="input-color">
                                    <input type="text" value={text} readOnly />
                                    <div className="color-box" style={style} />
                                </div>
                            </li>
                        );
                    })}
                </ul>
            </div>
        );
    }
}
ColonyFeatures.propTypes = {
    data: PropTypes.object.isRequired,
};
