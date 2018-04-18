import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import CCCRoot from '../../src/components/CCCRoot';
import cccMetadata from '../fixtures/cccMetadata';


describe('<CCCRoot />', () => {
    const onInitializeCCC = jasmine.createSpy('onInitialzeCCC');
    const onFinalizeCCC = jasmine.createSpy('onFinalizeCCC');
    const onError = jasmine.createSpy('onError');
    const props = { onError, onInitializeCCC, onFinalizeCCC };

    it('should render the error', () => {
        const error = 'Bad! Wrong! Boooh!';
        const wrapper = shallow(<CCCRoot {...props} error={error} />);
        expect(wrapper.text()).toContain(error);
    });

    describe('initializing', () => {
        it('should render a <CCCInitializationContainer />', () => {
            const wrapper = shallow(<CCCRoot {...props} />);
            expect(wrapper.find('CCCInitializationContainer').exists())
                .toBeTruthy();
        });

        it('should pass onInitializeCCC to <CCCInitialization />', () => {
            const wrapper = shallow(<CCCRoot {...props} />);
            expect(wrapper.find('CCCInitializationContainer').prop('onInitialize'))
                .toBe(onInitializeCCC);
        });

        it('should pass onError to <CCCInitialization />', () => {
            const wrapper = shallow(<CCCRoot {...props} />);
            expect(wrapper.find('CCCInitializationContainer').prop('onError'))
                .toBe(onError);
        });
    });

    describe('editing', () => {
        it('should render a <CCCEditorContainer />', () => {
            const wrapper = shallow(<CCCRoot {...props} cccMetadata={cccMetadata} />);
            expect(wrapper.find('CCCEditorContainer').exists()).toBeTruthy();
        });

        it('should pass cccMetadata to <CCCEditorContainer />', () => {
            const wrapper = shallow(<CCCRoot {...props} cccMetadata={cccMetadata} />);
            expect(wrapper.find('CCCEditorContainer').prop('cccMetadata'))
                .toEqual(cccMetadata);
        });

        it('should pass onFinalizeCCC to <CCCEditorContainer />', () => {
            const wrapper = shallow(<CCCRoot {...props} cccMetadata={cccMetadata} />);
            expect(wrapper.find('CCCEditorContainer').prop('onFinalizeCCC'))
                .toEqual(onFinalizeCCC);
        });
    });

    describe('finalized', () => {
        it('should render a <FinalizedCCC />', () => {
            const wrapper = shallow(<CCCRoot {...props} cccMetadata={cccMetadata} finalized />);
            expect(wrapper.find('FinalizedCCC').exists()).toBeTruthy();
        });

        it('should pass cccMetadata to <FinalizedCCC />', () => {
            const wrapper = shallow(<CCCRoot {...props} cccMetadata={cccMetadata} finalized />);
            expect(wrapper.find('FinalizedCCC').prop('cccMetadata')).toEqual(cccMetadata);
        });
    });
});
