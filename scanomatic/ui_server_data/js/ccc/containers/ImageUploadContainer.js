import PropTypes from 'prop-types';
import React from 'react';

import ImageUpload from '../components/ImageUpload';
import { uploadImage } from '../helpers';

export default class ImageUploadContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = { image: null };
        this.handleImageChange = this.handleImageChange.bind(this);
        this.handleUploadError = this.handleUploadError.bind(this);
    }

    handleImageChange(image) {
        this.setState({ image });
        if (image) {
            const { cccId, fixture, token } = this.props;
            uploadImage(cccId, image, fixture, token)
                .then(id => this.props.onFinish({ id, name: image.name }))
                .catch(this.handleUploadError);
        }
    }

    handleUploadError(reason) {
        alert(`An error occured while uploading the image: ${reason}`);
    }

    render() {
        return (
            <ImageUpload
                image={this.state.image}
                onImageChange={this.handleImageChange}
            />
        );
    }
}

ImageUploadContainer.propTypes = {
    cccId: PropTypes.string.isRequired,
    fixture: PropTypes.string.isRequired,
    token: PropTypes.string.isRequired,
    onFinish: PropTypes.func.isRequired,
};
