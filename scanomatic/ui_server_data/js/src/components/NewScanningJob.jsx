import PropTypes from 'prop-types';
import React from 'react';
import SoMPropTypes from '../prop-types';
import DurationInput from './DurationInput';

export default function NewScanningJob(props) {
    return (
        <div className="row">
            <div className="col-md-6 col-md-offset-3">
                <div className="panel panel-default">
                    <div className="panel-heading">
                        New scan series
                    </div>
                    {props.error && (
                        <div className="alert alert-danger" role="alert">
                            {props.error}
                        </div>
                    )}
                    <div className="panel-body">
                        <div className="form-group">
                            <label>Name</label>
                            <input
                                className="name form-control"
                                value={props.name}
                                placeholder="Short description of content"
                                onChange={props.onNameChange}
                            />
                        </div>
                        <DurationInput
                            duration={props.duration}
                            onChange={props.onDurationChange}
                        />
                        <div className="form-group">
                            <label>Interval</label>
                            <div className="input-group">
                                <input
                                    className="interval form-control"
                                    type="number"
                                    value={props.interval}
                                    placeholder="Interval (minutes)"
                                    onChange={props.onIntervalChange}
                                />
                                <span className="input-group-addon" id="interval-unit">minutes</span>
                            </div>
                        </div>
                        <div className="form-group">
                            <label>Scanner</label>
                            <select
                                className="scanner form-control"
                                onChange={props.onScannerChange}
                                value={props.scannerId}
                            >
                                {props.scanners
                                    .sort((a, b) => {
                                        if (a.name < b.name) return -1;
                                        if (a.name > b.name) return 1;
                                        return 0;
                                    })
                                    .map(v => (
                                        <option key={v.name} value={v.identifier}>
                                            {v.name}
                                            {` (${v.power ? 'online' : 'offline'}, ${v.owned ? 'occupied' : 'free'})`}
                                        </option>
                                    ))}
                            </select>
                        </div>
                        <button className="btn btn-primary job-add" onClick={props.onSubmit}>
                            Add to jobs
                        </button>
                        <button className="btn cancel" onClick={props.onCancel}>
                            Cancel
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}

NewScanningJob.propTypes = {
    name: PropTypes.string,
    error: PropTypes.string,
    duration: PropTypes.number,
    scannerId: PropTypes.string,
    scanners: PropTypes.arrayOf(SoMPropTypes.scannerType),
    interval: PropTypes.number.isRequired,
    onNameChange: PropTypes.func.isRequired,
    onDurationChange: PropTypes.func.isRequired,
    onIntervalChange: PropTypes.func.isRequired,
    onScannerChange: PropTypes.func.isRequired,
    onSubmit: PropTypes.func.isRequired,
    onCancel: PropTypes.func.isRequired,
};

NewScanningJob.defaultProps = {
    name: '',
    error: null,
    scannerId: '',
    scanners: [],
    duration: null,
};
