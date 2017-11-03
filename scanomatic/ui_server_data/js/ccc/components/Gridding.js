import PropTypes from 'prop-types';
import React from 'react';

import PlateContainer from '../containers/PlateContainer';

export default class Gridding extends React.Component {
    constructor(props) {
        super(props);
        this.handleRowOffsetChange = this.handleRowOffsetChange.bind(this);
        this.handleColOffsetChange = this.handleColOffsetChange.bind(this);
    }

    handleColOffsetChange(event) {
        this.props.onColOffsetChange(parseInt(event.target.value));
    }

    handleRowOffsetChange(event) {
        this.props.onRowOffsetChange(parseInt(event.target.value));
    }

    render() {
        const success = this.props.status === 'success';
        const loading = this.props.status === 'loading';
        return (
            <div>
                <div className="row">
                    <div className="col-md-12">
                        <div>
                            <div className="alert">{this.props.alert}</div>
                            <div>
                                <PlateContainer
                                    accessToken={this.props.accessToken}
                                    cccId={this.props.cccId}
                                    imageId={this.props.imageId}
                                    plateId={this.props.plateId}
                                    grid={this.props.grid}
                                    selectedColony={this.props.selectedColony}
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
                                                    value={this.props.rowOffset}
                                                    onChange={this.handleRowOffsetChange}
                                                />
                                            </td>
                                        </tr>
                                        <tr>
                                            <td>Cols</td>
                                            <td>
                                                <input
                                                    className='col-offset'
                                                    type="number"
                                                    value={this.props.colOffset}
                                                    onChange={this.handleColOffsetChange}
                                                />
                                            </td>
                                        </tr>
                                        </tbody>
                                    </table>
                                <button
                                    className="btn btn-default btn-xs btn-regrid"
                                    onClick={this.props.onRegrid}
                                    disabled={loading}
                                >Re-grid</button>
                            </div>
                            {success &&
                                <div>
                                    <button
                                        className="btn btn-default btn-xs btn-next"
                                        onClick={this.props.onNext}
                                    >Next Step</button>
                                </div>
                            }
                        </div>
                    </div>
                </div>
            </div>
        );
    }
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
