import React from 'react';
import { shallow, mount } from 'enzyme';

import './enzyme-setup';
import PolynomialResultsPlotScatter from '../../src/components/PolynomialResultsPlotScatter';

describe('<PolynomialResultsPlotScatter />', () => {
    const props = {
        resultsData: {
            calculated: [1, 2, 3, 4, 5],
            independentMeasurements: [2, 2, 3, 4, 5],
        },
        correlation: {
            slope: 4,
            intercept: 10,
            stderr: 0.11,
        },
    };

    it('renders a div to place the plot in', () => {
        const wrapper = shallow(<PolynomialResultsPlotScatter {...props} />);
        expect(wrapper.find('div.poly-corr-chart').exists()).toBeTruthy();
    });

    it('renders the title', () => {
        const wrapper = shallow(<PolynomialResultsPlotScatter {...props} />);
        expect(wrapper.find('h4').exists()).toBeTruthy();
        expect(wrapper.find('h4').text())
            .toEqual('Population Size Correlation');
    });

    it('renders summary paragraph', () => {
        const { slope, intercept, stderr } = props.correlation;
        const wrapper = shallow(<PolynomialResultsPlotScatter {...props} />);
        expect(wrapper.find('p').exists()).toBeTruthy();
        expect(wrapper.find('p').text())
            .toEqual(`Correlation: y = ${slope.toFixed(2)}x + ${intercept.toFixed(0)} (standard error ${stderr})`);
    });

    it('plots the data', () => {
        const wrapper = mount(<PolynomialResultsPlotScatter {...props} />);
        const expected = '<svg width="520" height="500" style="overflow: hidden;">';
        const result = wrapper.find('div.poly-corr-chart').html();
        expect(result).toContain(expected);
    });
});
