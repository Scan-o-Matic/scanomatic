import PropTypes from 'prop-types';
import React from 'react';

import GriddingContainer from '../containers/GriddingContainer';
import ColonyEditorContainer from '../containers/ColonyEditorContainer';
import PlateProgress from './PlateProgress';

export default function PlateEditor(props) {
    let title = "Step 2: Gridding";
    if (props.step === 'colony') {
        title = 'Colony Detection'
    }
    const [nCols, nRows] = props.pinFormat;
    return (
        <div>
            <h3>{title}</h3>
            <div className="row">
                <div className="col-md-6">
                    <GriddingContainer
                        accessToken={props.accessToken}
                        cccId={props.cccId}
                        imageId={props.imageId}
                        plateId={props.plateId}
                        pinFormat={props.pinFormat}
                        onFinish={props.onGriddingFinish}
                        selectedColony={props.selectedColony}
                    />
                </div>
                <div className="col-md-6">
                    {props.step === 'colony' &&
                        <ColonyEditorContainer
                            accessToken={props.accessToken}
                            ccc={props.cccId}
                            image={props.imageId}
                            plate={props.plateId}
                            onFinish={props.onColonyFinish}
                            row={props.selectedColony.row}
                            col={props.selectedColony.col}
                        />
                    }
                </div>
            </div>
            {props.step === 'colony' &&
                <div className="row">
                    <div className="col-md-12">
                        <PlateProgress
                            now={nCols * props.selectedColony.row + props.selectedColony.col}
                            max={nCols * nRows}
                        />
                    </div>
                </div>
            }
        </div>
    );
}

PlateEditor.propTypes = {
    pinFormat: PropTypes.arrayOf(PropTypes.number).isRequired,
    accessToken: PropTypes.string.isRequired,
    cccId: PropTypes.string.isRequired,
    imageId: PropTypes.string.isRequired,
    plateId: PropTypes.string.isRequired,
    step: PropTypes.oneOf(['gridding', 'colony']),
    onGriddingFinish: PropTypes.func,
    onColonyFinish: PropTypes.func,
    selectedColony: PropTypes.shape({
        row: PropTypes.number,
        col: PropTypes.number,
    }),
}
