import PropTypes from 'prop-types';
import React from 'react';


export default function PolynomialResultsInfo(
    { onClearError, polynomial, error }
) {
    if (error) {
        return (
            <div className="alert alert-error alert-dismissible" role="alert">
                <button
                    type="button"
                    className="close"
                    aria-label="Close"
                    onClick={onClearError}
                >
                    <span aria-hidden="true">&times;</span>
                </button>
                <span
                    className="glyphicon glyphicon-exclamation-sign"
                    aria-hidden="true"
                >
                </span>
                <strong>Error:</strong>
                {error}
            </div>
        );
    return (
        <div className='results'>
            <h3>Cell Count Calibration Polynomial</h3>
            <ul className='list-group'>
                <li className='list-group-item'>
                    <h4 className='list-group-item-heading'>
                    Polynomial
                    </h4>
                    <PolynomialEquation
                        coefficients={polynomial.coefficients}
                    />
                </li>
                <li className='list-group-item'>
                    <h4 className='list-group-item-heading'>
                    Colonies included
                    </h4>
                    {polynomial.colonies} colonies
                </li>
            </ul>
        </div>
    );
}

PolynomialResultsInfo.propTypes = {
    onClearError: PropTypes.func.isRequired,
    polynomial: PropTypes.shape({
        power: PropTypes.number.isRequired,
        coefficients: PropTypes.array.isRequired,
        colonies: PropTypes.number.isRequired,
    }),
    error: PropTypes.string,
};

export function PolynomialEquation({ coefficients }) {
    let poly = [];
    const polyPower = coefficients.length - 1;
    coefficients.map((coeff, position) => {
        if (coeff != 0) {
            let power = polyPower - position;
            const x = power == 0 ? null : <span className='variable'>x</span>;
            const pwr = power == 0 || power == 1 ? null : <sup>{power}</sup>;
            poly.push(
                <span key={`power-${power}`}>
                <ScientificNotation value={coeff} precision={4} />
                {x}{pwr}
                </span>
            );
        }
    });

    for (let i=poly.length - 1; i > 0; i--) {
        poly.splice(i, 0, <span key={`plus-${i}`}> + </span>);
    }

    if (poly.length == 0) {
        poly = 0;
    }

    return (
        <p className='math'>
            <span className='variable'>y</span> = {poly}
        </p>
    );
}

PolynomialEquation.propTypes = {
    coefficients: PropTypes.array.isRequired,
};


export function ScientificNotation({ value, precision }) {
    if (value == 0) {
        return <span>{value.toPrecision(precision)}</span>;
    }
    let absValue = Math.abs(value);
    let fraction = absValue < 1 ? absValue * 1000 : absValue / 1000;
    let power = Math.round(Math.log10(fraction))*3;
    if (
        (absValue > 1 && power < 3) ||
        (power > -3 && absValue < 1) ||
        absValue == 1
    ) {
        return <span>{value.toPrecision(precision)}</span>;
    } else {
        return (
            <span>
                {(value / Math.pow(10, power)).toPrecision(precision)}
                &times;<span>10<sup>{power}</sup></span>
            </span>
        );
    }
}

ScientificNotation.propTypes = {
    value: PropTypes.number.isRequired,
    precision: PropTypes.number.isRequired,
};
