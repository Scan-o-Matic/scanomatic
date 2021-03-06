import PropTypes from 'prop-types';
import React from 'react';

export const pinningFormats = [
    { val: '', txt: '  --- EMPTY --  ' },
    { val: '96', txt: '96 (8 x 12)' },
    { val: '386', txt: '364 (16 x 24)' },
    { val: '1536', txt: '1536 (32 x 48)' },
    { val: '6144', txt: '6144 (62 x 96)' },
];

export default class PinningInputs extends React.Component {
    handleChange(idx, value) {
        const pinning = Array.from(this.props.pinning);
        pinning[idx] = value;
        this.props.onChange(pinning);
    }

    render() {
        const { error, pinning } = this.props;
        const pinnings = [];
        pinning.forEach((value, idx) => {
            const plate = idx + 1;
            pinnings.push((
                <div key="plate-{plate}" className="input-group">
                    <span className="input-group-addon">Plate {plate}</span>
                    <select
                        className="pinning form-control"
                        onChange={e => this.handleChange(idx, e.target.value)}
                        value={value || ''}
                        name="new-exp-pinning-plate-{plate}"
                    >
                        {pinningFormats
                            .map(p => (
                                <option key={p.val} value={p.val}>
                                    {p.txt}
                                </option>
                            ))}
                    </select>
                </div>
            ));
        });
        return (
            <div className={`form-group group-pinning ${error ? 'has-error' : ''}`}>
                <label className="control-label">Pinning</label>
                {pinnings}
                {error && (
                    <span className="help-block">
                        {error}
                    </span>
                )}
            </div>
        );
    }
}

PinningInputs.propTypes = {
    error: PropTypes.string,
    pinning: PropTypes.arrayOf(PropTypes.string).isRequired,
    onChange: PropTypes.func.isRequired,
};

PinningInputs.defaultProps = {
    error: null,
};
