import PropTypes from 'prop-types';
import React from 'react';


export default class PolyResults extends React.Component {

    polynomialAsJSX(coefficients) {
        let poly = [];
        const polyPower = coefficients.length - 1;
        coefficients.map((coeff, position) => {
            if (coeff != 0) {
                let power = polyPower - position;
                poly.push(
                    <span key={`power-${power}`}>
                    {toScientificPrecision(coeff, 4)}
                    <span className='variable'>x</span><sup>{power}</sup>
                    </span>
                );
            }
        });

        for (let i=poly.length - 1; i > 0; i--) {
            poly.splice(i, 0, <span key={`plus-${i}`}> + </span>);
        }

        return (
            <p className='math'>
                <span className='variable'>y</span> = {poly}
            </p>
        );
    }

    render() {
        const { handleClearError, polynomial, data, error } = this.props;
        if (error) {
            return (
                <div className="alert alert-error alert-dismissible" role="alert">
                    <button type="button" className="close" aria-label="Close" onClick={handleClearError}><span aria-hidden="true">&times;</span></button>
                    <span className="glyphicon glyphicon-exclamation-sign" aria-hidden="true"></span>
                    <strong>Error:</strong>
                    {error}
                </div>
            );
        } else if (polynomial == null) {
            return null;
        }
        const plotData = data.independentMeasurements.map((value, idx) => {
            return {
                type: 'dot',
                x: value,
                y: data.calculated[idx],
            };
        });
        const polynomialText = this.polynomialAsJSX(polynomial.coefficients);
        return (
            <div>
                <h3>Cell Count Calibration Polynomial</h3>
                <ul className='list-group'>
                    <li className='list-group-item'>
                        <h4 className='list-group-item-heading'>
                        Polynomial
                        </h4>
                        {polynomialText}
                    </li>
                    <li className='list-group-item'>
                        <h4 className='list-group-item-heading'>
                        Colonies included
                        </h4>
                        {data.independentMeasurements.length} colonies
                    </li>
                </ul>
            </div>
        );
    }
}

PolyResults.propTypes = {
    handleClearError: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
        power: PropTypes.number.isRequired,
        coefficients: PropTypes.array.isRequired,
    }),
    data: PropTypes.shape({
        calculated: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
    }),
    error: PropTypes.string,
};

export function toScientificPrecision(value, precision) {
    let absValue = Math.abs(value);
    let fraction = absValue < 1 ? absValue * 1000 : absValue / 1000;
    let power = Math.round(Math.log10(fraction))*3;
    if ((absValue > 1 && power < 3) || (power > -3 && absValue < 1) || absValue == 1) {
        return value.toPrecision(precision);
    } else {
        return (
            <span>
                {(value / Math.pow(10, power)).toPrecision(precision)}
                &times;<span>10<sup>{power}</sup></span>
            </span>
        );
    }
}
