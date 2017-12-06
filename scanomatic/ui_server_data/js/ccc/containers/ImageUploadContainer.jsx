import PropTypes from 'prop-types';
import React from 'react';

import ImageUpload from '../components/ImageUpload';
import { uploadImage } from '../helpers';
import CCCPropTypes from '../prop-types';

export default class ImageUploadContainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = { image: null, progress: null };
        this.handleImageChange = this.handleImageChange.bind(this);
        this.handleUploadError = this.handleUploadError.bind(this);
        this.handleUploadSuccess = this.handleUploadSuccess.bind(this);
    }

    handleImageChange(image) {
        this.setState({ image });
        if (image) {
            const { id: cccId, fixtureName, accessToken } = this.props.cccMetadata;
            uploadImage(cccId, image, fixtureName, accessToken, this.setProgress.bind(this))
                .then(this.handleUploadSuccess)
                .catch(this.handleUploadError);
        }
    }

    handleUploadSuccess(id) {
        this.props.onFinish({ id, name: this.state.image.name });
        this.setState({ image: null, progress: null });
    }

    handleUploadError(reason) {
        this.setState({ image: null, progress: null });
        alert(`An error occured while uploading the image: ${reason}`);
    }

    setProgress(now, max, text) {
        this.setState({ progress: { now, max, text } });
    }

    render() {
        return (
            <ImageUpload
                image={this.state.image}
                progress={this.state.progress}
                onImageChange={this.handleImageChange}
            />
        );
    }
}

ImageUploadContainer.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata.isRequired,
    onFinish: PropTypes.func.isRequired,
};
