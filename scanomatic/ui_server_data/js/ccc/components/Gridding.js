import PropTypes from 'prop-types';
import React from 'react';

import PlateContainer from '../containers/PlateContainer';

export default function Gridding(props) {
    const success = props.status === 'success';
    const loading = props.status === 'loading';
    return (
        <div>
            <div className="row">
                <div className="col-md-12">
                    <div>
                        <div className="alert">{props.alert}</div>
                        <div>
                            <PlateContainer
                                accessToken={props.accessToken}
                                cccId={props.cccId}
                                imageId={props.imageId}
                                plateId={props.plateId}
                                grid={props.grid}
                                selectedColony={props.selectedColony}
                            />
                        </div>
                        <div>
                            <label>Input an offset and try again</label>
                                <table>
                                    <tbody>
                                    <tr>
                                        <td>Rows:</td>
                                        <td>
                                            <input
                                                className='row-offset'
                                                type="number"
                                                value={props.rowOffset}
                                                onChange={event => props.onRowOffsetChange(event.target.value)}
                                            />
                                        </td>
                                    </tr>
                                    <tr>
                                        <td>Cols</td>
                                        <td>
                                            <input
                                                className='col-offset'
                                                type="number"
                                                value={props.colOffset}
                                                onChange={event => props.onColOffsetChange(event.target.value)}
                                            />
                                        </td>
                                    </tr>
                                    </tbody>
                                </table>
                            <button
                                className="btn btn-default btn-xs btn-regrid"
                                onClick={props.onRegrid}
                                disabled={loading}
                            >Re-grid</button>
                        </div>
                        {success &&
                            <div>
                                <button
                                    className="btn btn-default btn-xs btn-next"
                                    onClick={props.onNext}
                                >Next Step</button>
                            </div>
                        }
                    </div>
                </div>
            </div>
        </div>
    );
}

Gridding.propTypes = {
    alert: PropTypes.string,
    colOffset: PropTypes.number.isRequired,
    onColOffsetChange: PropTypes.func,
    onNext: PropTypes.func,
    onRegrid: PropTypes.func,
    onRowOffsetChange: PropTypes.func,
    rowOffset: PropTypes.number.isRequired,
    status: PropTypes.oneOf(['success', 'error', 'loading']).isRequired,
    cccId: PropTypes.string,
    imageId: PropTypes.string,
    plateId: PropTypes.string,
    accessToken: PropTypes.string,
    grid: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number))),
    selectedColony: PropTypes.shape({
        row: PropTypes.number,
        col: PropTypes.number,
    }),
};
