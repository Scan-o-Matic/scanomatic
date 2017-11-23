import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import Root from '../../ccc/components/Root';
import cccMetadata from '../fixtures/cccMetadata';


describe('<Root />', () => {
    const onInitializeCCC = jasmine.createSpy('onInitialzeCCC');
    const onError = jasmine.createSpy('onError');
    const props = { onError, onInitializeCCC };

    it('should render the error', () => {
        const error = 'Bad! Wrong! Boooh!';
        const wrapper = shallow(<Root {...props} error={error} />);
        expect(wrapper.text()).toContain(error);
    });

    describe('initializing', () => {
        it('should render a <CCCInitializationContainer />', () => {
            const wrapper = shallow(<Root {...props} />);
            expect(wrapper.find('CCCInitializationContainer').exists())
                .toBeTruthy();
        });

        it('should pass onInitializeCCC to <CCCInitialization />', () => {
            const wrapper = shallow(<Root {...props} />);
            expect(wrapper.find('CCCInitializationContainer').prop('onInitialize'))
                .toBe(onInitializeCCC);
        });

        it('should pass onError to <CCCInitialization />', () => {
            const wrapper = shallow(<Root {...props} />);
            expect(wrapper.find('CCCInitializationContainer').prop('onError'))
                .toBe(onError);
        });
    });

    describe('editing', () => {
        it('should render a <CCCEditorContainer />', () => {
            const wrapper = shallow(<Root {...props} cccMetadata={cccMetadata} />);
            expect(wrapper.find('CCCEditorContainer').exists()).toBeTruthy();
        });

        it('should pass cccMetadata to <CCCEditorContainer />', () => {
            const wrapper = shallow(<Root {...props} cccMetadata={cccMetadata} />);
            expect(wrapper.find('CCCEditorContainer').prop('cccMetadata'))
                .toEqual(cccMetadata);
        });
    });
});
