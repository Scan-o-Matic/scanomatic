import PropTypes from 'prop-types';
import React from 'react';

export default class ImageUpload extends React.Component {
    constructor(props) {
        super(props);
        this.handleFileChange = this.handleFileChange.bind(this);
    }

    handleFileChange(event) {
        const { target: { files } } = event;
        this.props.onImageChange(files[0]);
    }

    render() {
        return (
            <div className="ImageUpload">
                <h3>Process new image</h3>
                <input type="file" onChange={this.handleFileChange} />
            </div>
        );
    }
}

ImageUpload.propTypes = {
    onImageChange: PropTypes.func.isRequired,
    image: PropTypes.instanceOf(File),
};
