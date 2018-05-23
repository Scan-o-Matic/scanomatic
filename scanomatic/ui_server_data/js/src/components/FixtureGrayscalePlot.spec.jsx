import { mount } from 'enzyme';
import React from 'react';

import './enzyme-setup';

import FixtureGrayscalePlot from './FixtureGrayscalePlot';

describe('<FixtureGrayscalePlot />', () => {
    const props = {
        width: 400,
        height: 200,
        pixelValues: [0, 100, 200],
        referenceValues: [10, 23, 44],
    };
    let wrapper;

    describe('with analysis', () => {
        beforeEach(() => {
            wrapper = mount(<FixtureGrayscalePlot {...props} />);
        });

        it('renders an svg', () => {
            expect(wrapper.find('.grayscale-graph').exists()).toBeTruthy();
        });
    });

    describe('without analysis', () => {
        beforeEach(() => {
            wrapper = mount(<FixtureGrayscalePlot
                {...props}
                pixelValues={null}
                referenceValues={null}
            />);
        });

        it('renders an alert', () => {
            expect(wrapper.find('.alert').exists()).toBeTruthy();
            expect(wrapper.find('.alert').text()).toEqual('No grayscale detected');
        });
    });
});
