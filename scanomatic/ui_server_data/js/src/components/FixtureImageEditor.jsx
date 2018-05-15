import React from 'react';
import PropTypes from 'prop-types';

import FixtureImage from './FixtureImage';
import FixtureImageDetail from './FixtureImageDetail';


export default class FixtureImageEditor extends React.Component {
    constructor(props) {
        super(props);
        this.handleMouse = this.handleMouse.bind(this);
        this.state = {};
    }

    getGrayScaleGraph() {
        const { x, y } = this.props.grayScale;
        return <div className="grayscale-graph" />;
    }

    handleAreaStart(pos) {
        this.setState({ areaStart: pos });
    }

    handleAreaEnd(pos) {
        if (pos && this.state.areaStart) {
            this.props.onAddArea(this.state.areaStart, pos);
        }
        this.setState({ areaStart: null });
    }

    handleMouse(pos) {
        this.setState({ mouse: pos });
    }

    render() {
        const {
            grayScaleType, grayScale, plates,
            onSave, onReset,
        } = this.props;
        const canFinalize = grayScale && grayScale.valid && plates && plates.length > 0;
        return (
            <div className="fixture-image">
                <div className="row">
                    <div className="col-md-10">
                        <div className="input-group">
                            <div className="input-group-addon">Gray Scale</div>
                            <select className="form-control" value={grayScaleType}>
                                <option value="">System Default</option>
                            </select>
                        </div>
                    </div>
                </div>
                <div className="row">
                    <div className="col-md-10">
                        <FixtureImage
                            onAreaStart={this.handleAreaStart}
                            onMouse={this.handleMouse}
                            onAreaEnd={this.handleAreaEnd}
                        />
                    </div>
                    <div className="col-md-2">
                        <div className="fixture-image-instrucitons">
                            <h2>Instructions</h2>
                            <ol>
                                <li>Mark area containing the gray scale</li>
                                <li>Verify that the result from the gray scale analysis seems valid</li>
                                <li>Mark plate areas</li>
                            </ol>
                            Click on area to undo it.
                        </div>
                        {this.getGrayScaleGraph()}
                        <FixtureImageDetail center={this.state.mouse} />
                        <button className="btn button-default" onClick={onReset}>Reset Areas</button>
                        <button className="btn button-primary" onClick={onSave} disabled={canFinalize} >Finalize</button>
                    </div>
                </div>
            </div>
        );
    }
}

FixtureImageEditor.propTypes = {
    uri: PropTypes.string.isRequired,
    grayScaleType: PropTypes.string.isRequired,
    grayScale: PropTypes.shape({
        x: PropTypes.arrayOf(PropTypes.number),
        y: PropTypes.arrayOf(PropTypes.number),
        valid: PropTypes.bool.isRequired,
    }),
    plates: PropTypes.arrayOf(PropTypes.shape({
        x1: PropTypes.number.isRequired,
        y1: PropTypes.number.isRequired,
        x2: PropTypes.number.isRequired,
        y2: PropTypes.number.isRequired,
    })),
    onAddArea: PropTypes.func.isRequired,
    onReset: PropTypes.func.isRequired,
    onSave: PropTypes.func.isRequired,
};

FixtureImageEditor.defaultProps = {
    grayScale: null,
    plates: null,
};
