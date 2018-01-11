import PropTypes from 'prop-types';
import React from 'react';

import PlateEditorContainer from '../containers/PlateEditorContainer';
import ImageUploadContainer from '../containers/ImageUploadContainer';
import PolynomialConstructionContainer from
    '../containers/PolynomialConstructionContainer';
import CCCPropTypes from '../prop-types';
import CCCInfoBox from './CCCInfoBox';


export default function CCCEditor(props) {
    return (
        <div>
            <div className="row">
                <div className="col-md-6">
                    <h1>Initiated CCC</h1>
                    <CCCInfoBox cccMetadata={props.cccMetadata} />
                </div>
            </div>
            {props.plates.map((plate, i) => (
                <PlateEditorContainer
                    key={`${plate.imageId}:${plate.plateId}`}
                    {...plate}
                    cccMetadata={props.cccMetadata}
                    onFinish={props.onFinishPlate}
                    collapse={props.currentPlate !== i}
                />
            ))}
            <ImageUploadContainer
                cccMetadata={props.cccMetadata}
                onFinish={props.onFinishUpload}
            />
            <PolynomialConstructionContainer
                cccMetadata={props.cccMetadata}
                onFinalizeCCC={props.onFinalizeCCC}
            />
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
    onFinalizeCCC: PropTypes.func.isRequired,
    onFinishPlate: PropTypes.func.isRequired,
    onFinishUpload: PropTypes.func.isRequired,
};
