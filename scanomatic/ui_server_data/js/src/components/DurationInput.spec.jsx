import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import DurationInput from './DurationInput';

describe('<DuraionInput />', () => {
    it('renders a days input', () => {
        const wrapper = shallow(<DurationInput />);
        expect(wrapper.find('input.days').exists()).toBeTruthy();
    });

    it('renders a hours input', () => {
        const wrapper = shallow(<DurationInput />);
        expect(wrapper.find('input.hours').exists()).toBeTruthy();
    });

    it('renders a minutes input', () => {
        const wrapper = shallow(<DurationInput />);
        expect(wrapper.find('input.minutes').exists()).toBeTruthy();
    });

    it('displays value from props', () => {
        const wrapper = shallow(<DurationInput duration={(86400 + 7200 + 180) * 1000} />);
        expect(wrapper.find('input.days').prop('value')).toEqual(1);
        expect(wrapper.find('input.hours').prop('value')).toEqual(2);
        expect(wrapper.find('input.minutes').prop('value')).toEqual(3);
    });

    it('shows the error state', () => {
        const wrapper = shallow(<DurationInput error="Bad!" />);
        expect(wrapper.find('div.form-group').hasClass('has-error')).toBeTruthy();
    });

    it('displays the error', () => {
        const wrapper = shallow(<DurationInput error="Bad!" />);
        const block = wrapper.find('span.help-block');
        expect(block.exists()).toBeTruthy();
        expect(block.text()).toEqual('Bad!');
    });

    describe('actions', () => {
        const onChange = jasmine.createSpy('onChange');
        let wrapper;

        beforeEach(() => {
            onChange.calls.reset();
            wrapper = shallow(<DurationInput onChange={onChange} />);
        });

        describe('days', () => {
            it('updates days on change but dont call onChange if in focus', () => {
                let input = wrapper.find('input.days');
                input.simulate('focus');
                input.simulate('change', { target: { value: 60 } });
                wrapper.update();
                input = wrapper.find('input.days');
                expect(input.prop('value')).toEqual(60);
                expect(onChange).not.toHaveBeenCalled();
            });

            it('calls onChange on days blur if duration changed', () => {
                let input = wrapper.find('input.days');
                input.simulate('change', { target: { value: 1 } });
                wrapper.update();
                input = wrapper.find('input.days');
                input.simulate('blur');
                expect(onChange).toHaveBeenCalledWith(86400000);
            });

            it('calls onChange on days change if not in focus', () => {
                const input = wrapper.find('input.days');
                input.simulate('change', { target: { value: 1 } });
                expect(onChange).toHaveBeenCalledWith(86400000);
            });
        });

        describe('hours', () => {
            it('updates hours on change but dont call onChange', () => {
                let input = wrapper.find('input.hours');
                input.simulate('focus');
                input.simulate('change', { target: { value: 30 } });
                wrapper.update();
                input = wrapper.find('input.hours');
                expect(input.prop('value')).toEqual(30);
                expect(onChange).not.toHaveBeenCalled();
            });

            it('calls onChange on hours blur if duration changed', () => {
                let input = wrapper.find('input.hours');
                input.simulate('change', { target: { value: 2 } });
                wrapper.update();
                input = wrapper.find('input.hours');
                input.simulate('blur');
                expect(onChange).toHaveBeenCalledWith(7200000);
            });

            it('calls onChange on hours change if not in focus', () => {
                const input = wrapper.find('input.hours');
                input.simulate('change', { target: { value: 2 } });
                expect(onChange).toHaveBeenCalledWith(7200000);
            });
        });

        describe('minutes', () => {
            it('updates minutes on change but dont call onChange if in focus', () => {
                let input = wrapper.find('input.minutes');
                input.simulate('focus');
                input.simulate('change', { target: { value: 3000 } });
                wrapper.update();
                input = wrapper.find('input.minutes');
                expect(input.prop('value')).toEqual(3000);
                expect(onChange).not.toHaveBeenCalled();
            });

            it('calls onChange on minutes blur if duration changed', () => {
                let input = wrapper.find('input.minutes');
                input.simulate('change', { target: { value: 3 } });
                wrapper.update();
                input = wrapper.find('input.minutes');
                input.simulate('blur');
                expect(onChange).toHaveBeenCalledWith(180000);
            });

            it('calls onChange on minutes change if not in focus', () => {
                const input = wrapper.find('input.minutes');
                input.simulate('change', { target: { value: 3 } });
                expect(onChange).toHaveBeenCalledWith(180000);
            });
        });

        it('sets values accoring to new props when change appears to come from the component', () => {
            const input = wrapper.find('input.minutes');
            input.simulate('focus');
            input.simulate('change', { target: { value: 3000 } });
            input.simulate('blur');
            wrapper.setProps({ duration: (86400 + 7200 + 180) * 1000 });
            wrapper.update();
            expect(wrapper.find('input.days').prop('value')).toEqual(1);
            expect(wrapper.find('input.hours').prop('value')).toEqual(2);
            expect(wrapper.find('input.minutes').prop('value')).toEqual(3);
        });

        it('keeps entered values if new props while inputting data', () => {
            let input = wrapper.find('input.minutes');
            input.simulate('focus');
            input.simulate('change', { target: { value: 3000 } });
            wrapper.update();
            input = wrapper.find('input.minutes');
            expect(input.prop('value')).toEqual(3000);
            wrapper.setProps({ duration: 0 });
            wrapper.update();
            expect(wrapper.find('input.minutes').prop('value')).toEqual(3000);
        });

        it('calls onChange with all changes (focus => change, change other field)', () => {
            let input = wrapper.find('input.minutes');
            input.simulate('focus');
            input.simulate('change', { target: { value: 10 } });
            wrapper.update();
            input = wrapper.find('input.hours');
            input.simulate('change', { target: { value: 2 } });
            expect(onChange).toHaveBeenCalledWith(7800000);
        });

        it('updates the inputs with all changes (focus => change, change other field)', () => {
            let input = wrapper.find('input.minutes');
            input.simulate('focus');
            input.simulate('change', { target: { value: 61 } });
            wrapper.update();
            input = wrapper.find('input.hours');
            input.simulate('change', { target: { value: 2 } });
            wrapper.update();
            const duration = (7200 + 3660) * 1000;
            wrapper.setProps({ duration });
            expect(wrapper.find('input.days').prop('value')).toEqual('');
            expect(wrapper.find('input.hours').prop('value')).toEqual(3);
            expect(wrapper.find('input.minutes').prop('value')).toEqual(1);
        });
    });
});
