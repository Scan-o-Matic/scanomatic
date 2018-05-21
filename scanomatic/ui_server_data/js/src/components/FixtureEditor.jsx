import React from 'react';
import PropTypes from 'prop-types';

import FixtureImage from './FixtureImage';
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
            imageUri, editActions, markers, areas,
        } = this.props;
        const { hover } = this.state;
        return (
            <div className="fixture-editor">
                <div className="fixture-editor-main">
                    <FixtureImage
                        onMouse={this.handleOnMouse}
                        imageUri={imageUri}
                        areas={areas}
                        markers={markers}
                        {...editActions}
                    />
                </div>
                <div className="fixture-editor-sidebar">
                    <ImageDetail
                        imageUri={imageUri}
                        width={DETAIL_SIZE}
                        height={DETAIL_SIZE}
                        {...hover}
                        crossHair
                    />
                </div>
            </div>
        );
    }
}

FixtureEditor.propTypes = {
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
    editActions: PropTypes.shape({
        onAreaStart: PropTypes.func.isRequired,
        onAreaEnd: PropTypes.func.isRequired,
        onClick: PropTypes.func.isRequired,
    }).isRequired,
};
