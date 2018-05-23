import React from 'react';
import PropTypes from 'prop-types';

import FixtureImage from './FixtureImage';
import FixtureGrayscalePlot from './FixtureGrayscalePlot';
import ImageDetail from './ImageDetail';

const DETAIL_SIZE = 201;

export default class FixtureEditor extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hover: { x: null, y: null } };
        this.handleOnMouse = this.handleOnMouse.bind(this);
    }

    handleOnMouse(hover) {
        if (hover == null) {
            this.setState({ hover: { x: null, y: null } });
        } else {
            this.setState({ hover });
        }
    }

    render() {
        const {
            imageUri, editActions, markers, areas, grayscaleType, scannerName, grayscaleDetection,
            validFixture, onFinalize, onResetAreas,
        } = this.props;
        const { hover } = this.state;
        return (
            <div className="fixture-editor">
                <h2>{scannerName}</h2>
                <div className="fixture-editor-flowrow">
                    <div className="fixture-editor-main">
                        <div className="form-group">
                            <label htmlFor="grayscale-type-select" className="control-label">Select Gray Scale</label>
                            <select className="form-control custom-inline-group grayscale-type" value={grayscaleType} id="grayscale-type-select" disabled>
                                <option value="silverfast">SilverFast</option>
                            </select>
                            <span className="help-block">Support for different types of grayscales implemented upon request</span>
                        </div>
                        <FixtureImage
                            onMouse={this.handleOnMouse}
                            imageUri={imageUri}
                            areas={areas}
                            markers={markers}
                            {...editActions}
                        />
                    </div>
                    <div className="fixture-editor-sidebar">
                        <h3>Instructions</h3>
                        <ol>
                            <li>Click - Drag - Release to mark gray scale area</li>
                            <li>When analysis is returned, verify it</li>
                            <li>Mark all plate areas</li>
                        </ol>
                        <p>
                            Click inside area to undo it or click reset all below to to remove all
                        </p>
                        <ImageDetail
                            imageUri={imageUri}
                            width={DETAIL_SIZE}
                            height={DETAIL_SIZE}
                            {...hover}
                            crossHair
                        />
                        {
                            (
                                grayscaleDetection
                                && <FixtureGrayscalePlot height={180} {...grayscaleDetection} />
                            )
                            || <FixtureGrayscalePlot />
                        }
                        <div className="flex-spacer" />
                        <button className="btn sidebar-btn reset-all-button" onClick={onResetAreas} disabled={areas.length === 0}>Reset all areas</button>
                        <button className="btn btn-primary sidebar-btn finalize-button" onClick={onFinalize} disabled={!validFixture}>
                            Finalize
                        </button>
                    </div>
                </div>
            </div>
        );
    }
}

FixtureEditor.propTypes = {
    areas: PropTypes.arrayOf(PropTypes.shape({
        name: PropTypes.string.isRequired,
        rect: PropTypes.shape({
            x: PropTypes.number.isRequired,
            y: PropTypes.number.isRequired,
            w: PropTypes.number.isRequired,
            h: PropTypes.number.isRequired,
        }).isRequired,
    })).isRequired,
    editActions: PropTypes.shape({
        onAreaStart: PropTypes.func.isRequired,
        onAreaEnd: PropTypes.func.isRequired,
        onClick: PropTypes.func.isRequired,
    }).isRequired,
    grayscaleDetection: PropTypes.shape({
        referenceValues: PropTypes.arrayOf(PropTypes.number).isRequired,
        pixelValues: PropTypes.arrayOf(PropTypes.number).isRequired,
    }),
    grayscaleType: PropTypes.string,
    imageUri: PropTypes.string.isRequired,
    markers: PropTypes.arrayOf(PropTypes.shape({
        x: PropTypes.number.isRequired,
        y: PropTypes.number.isRequired,
    })).isRequired,
    onFinalize: PropTypes.func.isRequired,
    onResetAreas: PropTypes.func.isRequired,
    scannerName: PropTypes.string.isRequired,
    validFixture: PropTypes.bool,
};

FixtureEditor.defaultProps = {
    grayscaleType: 'silverfast',
    grayscaleDetection: null,
    validFixture: false,
};
