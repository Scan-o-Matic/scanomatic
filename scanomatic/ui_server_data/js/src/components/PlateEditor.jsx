import PropTypes from 'prop-types';
import React from 'react';

import ColonyEditorContainer from '../containers/ColonyEditorContainer';
import PlateProgress from './PlateProgress';
import PlateContainer from '../containers/PlateContainer';
import Gridding from './Gridding';
import CCCPropTypes from '../prop-types';


const STEPS = ['pre-processing', 'gridding', 'colony-detection', 'done'];


export function PlateStatusLabel({ step, griddingError, now, max }) {
    let className = 'pull-right label'
    let text = '';
    if (step === 'pre-processing') {
        text = 'Pre-processing...';
        className += ' label-default';
    }
    else if (step === 'gridding' && !griddingError) {
        text = 'Gridding...';
        className += ' label-default';
    }
    else if (step === 'gridding') {
        text = 'Gridding error';
        className += ' label-danger';
    } else if (step === 'colony-detection') {
        text = `${now}/${max}`;
        className += ' label-primary';
    } else if (step === 'done') {
        text = 'Done!';
        className += ' label-success';
    }

    return <span className={className}>{text}</span>
}

PlateStatusLabel.propTypes = {
    griddingError: PropTypes.string,
    now: PropTypes.number,
    max: PropTypes.number,
    step: PropTypes.oneOf(STEPS).isRequired,
}

export default function PlateEditor(props) {
    let title = "Step 2: Gridding";
    if (props.step === 'colony-detection') {
        title = 'Step 3: Colony Detection'
    }
    const { nCols, nRows } = props.cccMetadata.pinningFormat;
    const now = nCols * props.selectedColony.row + props.selectedColony.col;
    const max = nCols * nRows;
    return (
        <div className="panel panel-default">
            <div className="panel-heading">
                <PlateStatusLabel
                    step={props.step}
                    now={now}
                    max={max}
                    griddingError={props.griddingError}
                />
               {props.imageName}, Plate {props.plateId}
            </div>
            <div
                className={props.collapse ? "panel-body collapse" : "panel-body"}
            >
                <h3>{title}</h3>
                <div className="row">
                    <div className="col-md-6 text-center">
                        <PlateContainer
                            cccId={props.cccMetadata.id}
                            imageId={props.imageId}
                            plateId={props.plateId}
                            selectedColony={props.selectedColony}
                            grid={props.grid}
                        />
                    </div>
                    <div className="col-md-6">
                        {props.step === 'gridding' &&
                            <div className="well">
                                <Gridding
                                    rowOffset={props.rowOffset}
                                    colOffset={props.colOffset}
                                    onRowOffsetChange={props.onRowOffsetChange}
                                    onColOffsetChange={props.onColOffsetChange}
                                    onRegrid={props.onRegrid}
                                    error={props.griddingError}
                                    loading={props.griddingLoading}
                                />
                            </div>
                        }
                        {props.step === 'colony-detection' &&
                            <ColonyEditorContainer
                                accessToken={props.cccMetadata.accessToken}
                                ccc={props.cccMetadata.id}
                                image={props.imageId}
                                plateId={props.plateId}
                                onFinish={props.onColonyFinish}
                                row={props.selectedColony.row}
                                col={props.selectedColony.col}
                            />
                        }
                    </div>
                </div>
                <div className="row">
                    <div className="col-md-12 text-right">
                        {props.step === 'gridding' &&
                            <button
                                className="btn btn-primary btn-next"
                                disabled={!!props.griddingError || props.griddingLoading}
                                onClick={props.onClickNext}
                            >Next</button>
                        }
                        {props.step === 'colony-detection' &&
                            <button className="btn btn-success"
                                onClick={props.onClickNext}
                            >Done</button>
                        }
                    </div>
                </div>
                {props.step === 'colony-detection' &&
                    <div className="row">
                        <div className="col-md-12">
                            <PlateProgress now={now} max={max} />
                        </div>
                    </div>
                }
            </div>
        </div>

    );
}

PlateEditor.propTypes = {
    cccMetadata: CCCPropTypes.cccMetadata.isRequired,
    imageId: PropTypes.string.isRequired,
    imageName: PropTypes.string.isRequired,
    plateId: PropTypes.number.isRequired,
    step: PropTypes.oneOf(STEPS).isRequired,
    grid: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number))),
    griddingLoading: PropTypes.bool,
    griddingError: PropTypes.string,
    onClickNext: PropTypes.func,
    onColonyFinish: PropTypes.func,
    selectedColony: PropTypes.shape({
        row: PropTypes.number,
        col: PropTypes.number,
    }),
    rowOffset: PropTypes.number.isRequired,
    colOffset: PropTypes.number.isRequired,
    onRowOffsetChange: PropTypes.func,
    onColOffsetChange: PropTypes.func,
    onRegrid: PropTypes.func,
    collapse: PropTypes.bool,
}
