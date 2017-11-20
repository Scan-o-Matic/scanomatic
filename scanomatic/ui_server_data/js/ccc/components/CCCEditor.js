import PropTypes from 'prop-types';
import React from 'react';

import PlateEditorContainer from '../containers/PlateEditorContainer';
import ImageUploadContainer from '../containers/ImageUploadContainer';
import PolynomialConstructionContainer from
    '../containers/PolynomialConstructionContainer';


export default function CCCEditor(props) {
    return (
        <div>
            {props.plates.map((plate, i) =>
                <PlateEditorContainer key={i}
                    {...plate}
                    pinFormat={props.pinFormat}
                    accessToken={props.accessToken}
                    cccId={props.cccId}
                    onFinish={props.onFinishPlate}
                    collapse={props.currentPlate != i}
                />
            )}
            <ImageUploadContainer
                cccId={props.cccId}
                token={props.accessToken}
                fixture={props.fixtureName}
                onFinish={props.onFinishUpload}
            />
            <PolynomialConstructionContainer
                cccId={props.cccId}
                accessToken={props.accessToken}
            />
        </div>
    );
}

CCCEditor.propTypes = {
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    plates: PropTypes.arrayOf(PropTypes.shape({
        imageId: PropTypes.string.isRequired,
        imageName: PropTypes.string.isRequired,
        plateId: PropTypes.number.isRequired,
    })).isRequired,
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
    currentPlate: PropTypes.number,
    fixtureName: PropTypes.string.isRequired,
    onFinishPlate: PropTypes.func.isRequired,
    onFinishUpload: PropTypes.func.isRequired,
};
