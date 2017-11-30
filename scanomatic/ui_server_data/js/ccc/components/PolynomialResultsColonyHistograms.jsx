import PropTypes from 'prop-types';
import React from 'react';
import PolynomialResultsColonyHistogram from './PolynomialResultsColonyHistogram';


const PolynomialResultsColonyHistograms = (props) => {
    const colonies = [];
    const {
        independentMeasurements, pixelValues, pixelCounts, maxCount,
        minPixelValue, maxPixelValue,
    } = props.colonies;

    for (let i = 0; i < pixelValues.length; i += 1) {
        colonies.push((
            <PolynomialResultsColonyHistogram
                independentMeasurement={independentMeasurements[i]}
                pixelValues={pixelValues[i]}
                pixelCounts={pixelCounts[i]}
                maxCount={maxCount}
                maxPixelValue={maxPixelValue}
                minPixelValue={minPixelValue}
                colonyIdx={i}
                key={`colony-${i}`}
            />
        ));
    }
    return (
        <div className="poly-histograms">
            <h4>Colony Histograms</h4>
            {colonies}
        </div>
    );
};

PolynomialResultsColonyHistograms.propTypes = {
    colonies: PropTypes.shape({
        pixelValues: PropTypes.array.isRequired,
        pixelCounts: PropTypes.array.isRequired,
        independentMeasurements: PropTypes.array.isRequired,
        maxCount: PropTypes.number.isRequired,
        maxPixelValue: PropTypes.number.isRequired,
        minPixelValue: PropTypes.number.isRequired,
    }).isRequired,
};

export default PolynomialResultsColonyHistograms;
