import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import PolynomialResultsColonyHistograms from
    '../../src/components/PolynomialResultsColonyHistograms';

describe('<PolynomialResultsColonyHistograms />', () => {
    const props = {
        colonies: {
            pixelValues: [[1, 2], [5.5]],
            pixelCounts: [[100, 1], [44]],
            independentMeasurements: [123, 441],
            minPixelValue: 1,
            maxPixelValue: 5.5,
            maxCount: 100,
        },
    };

    it('Renders a header', () => {
        const wrapper = shallow(<PolynomialResultsColonyHistograms {...props} />);
        expect(wrapper.find('h4').exists()).toBeTruthy();
        expect(wrapper.find('h4').text()).toContain('Colony Histograms');
    });

    it('Renders the expected number of histograms', () => {
        const wrapper = shallow(<PolynomialResultsColonyHistograms {...props} />);
        expect(wrapper.find('PolynomialResultsColonyHistogram').length).toEqual(2);
    });

    it('Passes the data to the histograms', () => {
        const wrapper = shallow(<PolynomialResultsColonyHistograms {...props} />);
        const histograms = wrapper.find('PolynomialResultsColonyHistogram');
        for (let i = 0; i < histograms.length; i += 1) {
            const hist = histograms.get(i);
            expect(hist.props.pixelValues).toEqual(props.colonies.pixelValues[i]);
            expect(hist.props.pixelCounts).toEqual(props.colonies.pixelCounts[i]);
            expect(hist.props.independentMeasurement)
                .toEqual(props.colonies.independentMeasurements[i]);
            expect(hist.props.maxCount).toEqual(props.colonies.maxCount);
            expect(hist.props.maxPixelValue).toEqual(props.colonies.maxPixelValue);
            expect(hist.props.minPixelValue).toEqual(props.colonies.minPixelValue);
            expect(hist.props.colonyIdx).toEqual(i);
        }
    });
});
