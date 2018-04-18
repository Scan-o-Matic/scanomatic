import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import CCCInitialization from '../../src/components/CCCInitialization';


describe('<CCCInitialization />', () => {
    const onSpeciesChange = jasmine.createSpy('onSpeciesChange');
    const onReferenceChange = jasmine.createSpy('onReferenceChange');
    const onFixtureNameChange = jasmine.createSpy('onFixtureNameChange');
    const onPinningFormatNameChange = jasmine.createSpy('onPinningFormatNameChange');
    const onSubmit = jasmine.createSpy('onSubmit');
    const props = {
        species: 'S. Kombuchae',
        reference: 'Professor X',
        fixtureName: 'fix1',
        fixtureNames: ['fix0', 'fix1'],
        pinningFormatName: '2x4',
        pinningFormatNames: ['1x1', '2x4'],
        onSpeciesChange,
        onReferenceChange,
        onFixtureNameChange,
        onPinningFormatNameChange,
        onSubmit,
    };

    it('should render an <input /> for the species', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('input.species');
        expect(input.exists()).toBeTruthy();
        expect(input.prop('value')).toEqual(props.species);
    });

    it('should call onSpeciesChange when the species input changes', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('input.species');
        const event = { target: { value: 'XXX' } };
        input.simulate('change', event);
        expect(onSpeciesChange).toHaveBeenCalledWith(event);
    });

    it('should render an <input /> for the reference', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('input.reference');
        expect(input.exists()).toBeTruthy();
        expect(input.prop('value')).toEqual(props.reference);
    });

    it('should call onReferenceChange when the reference changes', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('input.reference');
        const event = { target: { value: 'XXX' } };
        input.simulate('change', event);
        expect(onReferenceChange).toHaveBeenCalledWith(event);
    });

    it('should render a <select /> for the fixtures', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('select.fixtures');
        expect(input.exists()).toBeTruthy();
        expect(input.prop('value')).toEqual(props.fixtureName);
        expect(input.find('option').at(0).text()).toEqual('fix0');
        expect(input.find('option').at(0).prop('value')).toEqual('fix0');
        expect(input.find('option').at(1).text()).toEqual('fix1');
        expect(input.find('option').at(1).prop('value')).toEqual('fix1');
    });

    it('should call onFixtureChange when the selected fixture changes', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('select.fixtures');
        const event = { target: { value: 'XXX' } };
        input.simulate('change', event);
        expect(onFixtureNameChange).toHaveBeenCalledWith(event);
    });

    it('should render a <select /> for the pinning formats', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('select.pinningformats');
        expect(input.exists()).toBeTruthy();
        expect(input.find('option').at(0).text()).toEqual('1x1');
        expect(input.find('option').at(0).prop('value')).toEqual('1x1');
        expect(input.find('option').at(1).text()).toEqual('2x4');
        expect(input.find('option').at(1).prop('value')).toEqual('2x4');
    });

    it('should call onPinningFormatNameChange when the selected pinning format changes', () => {
        const wrapper = shallow(<CCCInitialization {...props} />);
        const input = wrapper.find('select.pinningformats');
        const event = { target: { value: 'XXX' } };
        input.simulate('change', event);
        expect(onPinningFormatNameChange).toHaveBeenCalledWith(event);
    });
});
