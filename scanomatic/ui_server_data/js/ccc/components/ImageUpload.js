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
        const progressWidth = this.props.progress
            ? (this.props.progress.now / this.props.progress.max * 100)
            : 0;
        return (
            <div className="ImageUpload">
                <h3>Process new image</h3>
                {this.props.progress ?
                    <div>
                        <div className="progress">
                            <div
                                className="progress-bar"
                                style={{ width: `${progressWidth}%` }} />
                        </div>
                        {this.props.progress.text}
                    </div>
                :
                    <input
                        type="file"
                        onChange={this.handleFileChange}
                    />
                }
            </div>
        );
    }
}

ImageUpload.propTypes = {
    onImageChange: PropTypes.func.isRequired,
    image: PropTypes.instanceOf(File),
    progress: PropTypes.shape({
        now: PropTypes.number.isRequired,
        max: PropTypes.number.isRequired,
        text: PropTypes.string.isRequired,
    }),
};
