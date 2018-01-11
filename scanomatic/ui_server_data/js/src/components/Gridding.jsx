import PropTypes from 'prop-types';
import React from 'react';


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
        return (
            <div>
                <h4>Gridding</h4>
                {this.props.loading &&
                    <div className="progress">
                        <div
                            className="progress-bar progress-bar-striped active"
                            style={{ width: '100%' }}>
                        </div>
                    </div>
                }
                {!this.props.loading &&
                    <form>
                        {this.props.error &&
                            <div className="alert alert-danger" >
                                {this.props.error}
                            </div>
                        }
                        {!this.props.error &&
                            <div className="alert alert-success" >
                                Gridding was succesful!
                            </div>
                        }
                        <div className="form-group">
                            <label>Rows</label>
                            <input
                                className='row-offset form-control'
                                type="number"
                                value={this.props.rowOffset}
                                onChange={this.handleRowOffsetChange}
                            />
                        </div>
                        <div className="form-group">
                            <label>Columns</label>
                            <input
                                className='col-offset form-control'
                                type="number"
                                value={this.props.colOffset}
                                onChange={this.handleColOffsetChange}
                            />
                        </div>
                        <div className="text-right">
                            <button
                                className="btn btn-default btn-regrid"
                                onClick={this.props.onRegrid}
                            >Re-grid</button>
                        </div>
                    </form>
                }
            </div>
        );
    }
}

Gridding.propTypes = {
    error: PropTypes.string,
    loading: PropTypes.bool,
    rowOffset: PropTypes.number.isRequired,
    colOffset: PropTypes.number.isRequired,
    onRegrid: PropTypes.func,
    onRowOffsetChange: PropTypes.func,
    onColOffsetChange: PropTypes.func,
};
