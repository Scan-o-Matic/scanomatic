import PropTypes from 'prop-types';
import React from 'react';

import PlateEditorContainer from '../containers/PlateEditorContainer';
import ImageUploadContainer from '../containers/ImageUploadContainer';
import PolynomialConstructionContainer from
    '../containers/PolynomialConstructionContainer';
import CCCPropTypes from '../prop-types';


export default function CCCEditor(props) {
    return (
        <div>
            {props.plates.map((plate, i) =>
                <PlateEditorContainer key={i}
                    {...plate}
                    cccMetadata={props.cccMetadata}
                    onFinish={props.onFinishPlate}
                    collapse={props.currentPlate != i}
                />
            )}
            <ImageUploadContainer
                cccMetadata={props.cccMetadata}
                onFinish={props.onFinishUpload}
            />
            <PolynomialConstructionContainer cccMetadata={props.cccMetadata} />
        </div>
    );
}

CCCEditor.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata.isRequired,
    plates: PropTypes.arrayOf(PropTypes.shape({
        imageId: PropTypes.string.isRequired,
        imageName: PropTypes.string.isRequired,
        plateId: PropTypes.number.isRequired,
    })).isRequired,
    currentPlate: PropTypes.number,
    onFinishPlate: PropTypes.func.isRequired,
    onFinishUpload: PropTypes.func.isRequired,
};
