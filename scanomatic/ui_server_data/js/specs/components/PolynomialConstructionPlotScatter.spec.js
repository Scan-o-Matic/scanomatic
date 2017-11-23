import React from 'react';
import { shallow, mount } from 'enzyme';

import './enzyme-setup';
import PolynomialResultsPlotScatter from '../../ccc/components/PolynomialResultsPlotScatter';

describe('<PolynomialResultsPlotScatter />', () => {
    const props = {
        resultsData: {
            calculated: [1, 2, 3, 4, 5],
            independentMeasurements: [2, 2, 3, 4, 5],
        },
    };

    it('renders a div to place the plot in', () => {
        const wrapper = shallow(<PolynomialResultsPlotScatter {...props} />);
        expect(wrapper.find('div.poly-corr-chart').exists()).toBeTruthy();
    });

    it('plots the data', () => {
        const wrapper = mount(<PolynomialResultsPlotScatter {...props} />);
        expect(wrapper.find('div.poly-corr-chart').html())
            .toContain('<svg');
    });
});
