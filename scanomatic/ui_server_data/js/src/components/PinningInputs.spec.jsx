import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';

import PinningInputs, { pinningFormats } from './PinningInputs';

describe('<PinningInputs />', () => {
    const onChange = jasmine.createSpy('onChange');
    const props = {
        pinning: new Map([[1, null], [2, '1536']]),
        onChange,
    };

    beforeEach(() => {
        onChange.calls.reset();
    });

    describe('no error', () => {
        let wrapper;

        beforeEach(() => {
            wrapper = shallow(<PinningInputs {...props} />);
        });

        it('is a form-group without erros', () => {
            expect(wrapper.hasClass('form-group')).toBeTruthy();
            expect(wrapper.hasClass('has-error')).toBeFalsy();
        });

        it('has a label', () => {
            const label = wrapper.find('.control-label');
            expect(label.exists()).toBeTruthy();
            expect(label.text()).toEqual('Pinning');
        });

        it('has no help-block', () => {
            expect(wrapper.find('.help-block').exists()).toBeFalsy();
        });

        describe('inputs', () => {
            let inputs;

            beforeEach(() => {
                inputs = wrapper.find('.input-group');
            });

            it('renders one per key in pinning Map', () => {
                expect(inputs.exists()).toBeTruthy();
                expect(inputs.length).toEqual(2);
            });

            it('sets empty string value of the first select', () => {
                const select = inputs.at(0).find('select');
                expect(select.exists()).toBeTruthy();
                expect(select.prop('value')).toEqual('');
            });

            it('sets the value of the second select', () => {
                const select = inputs.at(1).find('select');
                expect(select.exists()).toBeTruthy();
                expect(select.prop('value')).toEqual('1536');
            });

            it('sets the group addon label of the first select', () => {
                const addon = inputs.at(0).find('.input-group-addon');
                expect(addon.exists()).toBeTruthy();
                expect(addon.text()).toEqual('Plate 1');
            });

            it('sets the group addon label of the second select', () => {
                const addon = inputs.at(1).find('.input-group-addon');
                expect(addon.exists()).toBeTruthy();
                expect(addon.text()).toEqual('Plate 2');
            });

            it('should call onChange with updated pinning on change', () => {
                const evt = { target: { value: '384' } };
                const select = inputs.at(0).find('select');
                select.simulate('change', evt);
                expect(onChange).toHaveBeenCalledWith(new Map([[1, '384'], [2, '1536']]));
            });

            it('should set the correct values of the expected options of the selects', () => {
                const select = inputs.at(0);
                expect(select.find('option').map(e => e.prop('value')))
                    .toEqual(pinningFormats.map(e => e.val));
            });

            it('should render the correct texts of the expected options of the selects', () => {
                const select = inputs.at(0);
                expect(select.find('option').map(e => e.text()))
                    .toEqual(pinningFormats.map(e => e.txt));
            });
        });
    });

    describe('with error', () => {
        let wrapper;
        const error = 'Bad bad bad!!';

        beforeEach(() => {
            wrapper = shallow(<PinningInputs {...props} error={error} />);
        });

        it('is a form-group with erros', () => {
            expect(wrapper.hasClass('has-error')).toBeTruthy();
        });

        it('has help-block', () => {
            const block = wrapper.find('.help-block');
            expect(block.exists()).toBeTruthy();
            expect(block.text()).toEqual(error);
        });
    });
});
