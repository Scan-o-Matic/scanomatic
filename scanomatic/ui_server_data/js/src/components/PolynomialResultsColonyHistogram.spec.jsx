import React from 'react';
import { shallow, mount } from 'enzyme';

import './enzyme-setup';
import PolynomialResultsColonyHistogram from '../../src/components/PolynomialResultsColonyHistogram';

describe('<PolynomialResultsPlotScatter />', () => {
    const props = {
        colonyIdx: 5,
        pixelValues: [2.4, 5.3, 12.2],
        pixelCounts: [3, 15, 5],
        independentMeasurement: 424242,
        maxPixelValue: 14,
        minPixelValue: 1.2,
        maxCount: 2003,
    };

    it('renders an container div', () => {
        const wrapper = shallow(<PolynomialResultsColonyHistogram {...props} />);
        expect(wrapper.find('div.poly-colony-container').exists()).toBeTruthy();
    });

    it('renders a div to place the plot in', () => {
        const wrapper = shallow(<PolynomialResultsColonyHistogram {...props} />);
        expect(wrapper.find('div.poly-colony-chart').exists()).toBeTruthy();
    });

    it('renders the population size', () => {
        const wrapper = shallow(<PolynomialResultsColonyHistogram {...props} />);
        expect(wrapper.find('span.poly-colony-txt').exists()).toBeTruthy();
        expect(wrapper.find('span.poly-colony-txt').text())
            .toEqual('4.24 x 10^5 cells');
    });

    it('plots the data', () => {
        const wrapper = mount(<PolynomialResultsColonyHistogram {...props} />);
        const expected = '<svg width="520" height="100" style="overflow: hidden;">';
        const result = wrapper.find('div.poly-colony-chart').html();
        expect(result).toContain(expected);
    });
});
