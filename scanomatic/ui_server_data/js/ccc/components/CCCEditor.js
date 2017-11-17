import PropTypes from 'prop-types';
import React from 'react';

import PlateEditorContainer from '../containers/PlateEditorContainer';
import ImageUploadContainer from '../containers/ImageUploadContainer';
import PlateList from './PlateList';

export default function CCCEditor(props) {
    let view;

    if (props.currentImage == null) {
        view = (
            <ImageUploadContainer
                cccId={props.cccId}
                token={props.accessToken}
                fixture={props.fixtureName}
                onFinish={props.onFinishUpload}
            />
        );
    } else {
        const image = props.images[props.currentImage];
        view = (
            <PlateEditorContainer
                pinFormat={props.pinFormat}
                accessToken={props.accessToken}
                cccId={props.cccId}
                imageId={image.id}
                imageName={image.name}
                plateId={1}
                onFinish={props.onFinishImage}
            />
        );
    }

    return (
        <div>
            <PlateList plates={props.images} />
            {view}
        </div>
    );
}

CCCEditor.propTypes = {
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    images: PropTypes.arrayOf(PropTypes.shape({
        name: PropTypes.string.isRequired,
        id: PropTypes.string.isRequired,
    })).isRequired,
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
    currentImage: PropTypes.number,
    fixtureName: PropTypes.string.isRequired,
    onFinishImage: PropTypes.func,
    onFinishUpload: PropTypes.func,
};
