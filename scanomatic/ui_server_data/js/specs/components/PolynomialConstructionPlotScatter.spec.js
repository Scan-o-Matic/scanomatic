import React from 'react';
import { shallow, mount } from 'enzyme';

import './enzyme-setup';
import PolynomialResultsPlotScatter, { labelFormatter } from '../../ccc/components/PolynomialResultsPlotScatter';

const toLookLikeSVG = (util, customEqualityTesters) => ({
    compare: (actual, expected) => {
        const expectedData = expected.match(/<svg.*<\/svg>/)[0];
        const expectedDataURL = `data://text/svg,${expectedData}`;
        const actualData = actual.match(/<svg.*<\/svg>/)[0];
        const actualDataURL = `data://text/svg,${actualData}`;
        const result = {
            pass: util.equals(actual, expected, customEqualityTesters),
        };
        result.message = result.pass ? 'Expected SVGs to look different' : 'Expected SVGs to look the same';
        result.message += `\n\texpected: ${expectedDataURL}`;
        result.message += `\n\tactual: " ${actualDataURL}`;
        return result;
    },
});

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
        }
    };

    beforeEach(() => {
        jasmine.addMatchers({ toLookLikeSVG });
    });

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
        const clock = jasmine.clock().install();
        clock.mockDate(new Date(42));
        const wrapper = mount(<PolynomialResultsPlotScatter {...props} />);
        const expected = '<svg width="520" height="500" style="overflow: hidden;">';
        const result = wrapper.find('div.poly-corr-chart').html();
        expect(result).toContain(expected);
        jasmine.clock().uninstall();
    });
});

describe('labelFormatter', () => {
    it('returns zero', () => {
        expect(labelFormatter(0)).toEqual('0');
    });

    it('returns the expected output for value', () => {
        expect(labelFormatter(320)).toEqual('3 x 10^2');
    });

    it('respects fixed positions', () => {
        expect(labelFormatter(320, 1)).toEqual('3.2 x 10^2');
    });
});
