import PropTypes from 'prop-types';
import React from 'react';

import CCCEditor from '../components/CCCEditor';


export default class CCCEditorContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            images: [],
            currentImage: null,
        };
        this.handleFinishUpload = this.handleFinishUpload.bind(this);
        this.handleFinishImage = this.handleFinishImage.bind(this);
    }

    handleFinishUpload(newImage) {
        this.setState({
            images: [...this.state.images, newImage],
            currentImage: this.state.images.length,
        });
    }

    handleFinishImage() {
        this.setState({ currentImage: null });
    }

    render() {
        return (
            <CCCEditor
                cccId={this.props.cccId}
                accessToken={this.props.accessToken}
                pinFormat={this.props.pinFormat}
                fixtureName={this.props.fixtureName}
                images={this.state.images}
                currentImage={this.state.currentImage}
                onFinishUpload={this.handleFinishUpload}
                onFinishImage={this.handleFinishImage}
            />
        );
    }
}

CCCEditorContainer.propTypes = {
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    fixtureName: PropTypes.string.isRequired,
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
};
