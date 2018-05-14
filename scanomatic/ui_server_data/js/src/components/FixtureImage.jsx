import React from 'react';
import PropTypes from 'prop-types';

export default class FixtureImage extends React.Component {
    constructor(props) {
        super(props);
        this.handleMouseMove = this.handleMouseMove.bind(this);
    }

    handleMouseMove(e) {

    }

    getGrayScaleGraph() {
        return <div className="grayscale-graph" />;
    }

    render() {
        const {
            grayScaleType, grayScale, plates, uri,
            onSave, onReset, onChange,
        } = this.props;
        const canFinalize = grayScale && grayScale.valid && plates && plates.length > 0;
        const canEdit = onSave && onReset && onChange;
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
                        <img src={uri} alt="Full resolution view" />
                    </div>
                    {canEdit &&
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
                            <button className="btn button-primary" onClick={onSave} disabled={canFinalize} >Finalize</button>
                        </div>
                    }
                </div>
            </div>
        );
    }
}

FixtureImage.propTypes = {
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
    onChange: PropTypes.func,
    onReset: PropTypes.func,
    onSave: PropTypes.func,
};

FixtureImage.defaultProps = {
    grayScale: null,
    plates: null,
    onSave: null,
    onReset: null,
    onChange: null,
};
