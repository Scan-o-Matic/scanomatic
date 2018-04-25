import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import NewExperimentPanel from './NewExperimentPanel';

describe('<NewExperimentPanel/>', () => {
    const defaultProps = {
        project: 'I project',
        name: 'Cool idea',
        description: 'Test all the things',
        duration: 120000,
        interval: 60000,
        scanners: [
            {
                name: 'Tox',
                owned: false,
                power: true,
                identifier: 'hoho',
            },
            {
                name: 'Npm',
                owned: true,
                power: false,
                identifier: 'haha',
            },
        ],
        onChange: jasmine.createSpy('onNameChange'),
        onSubmit: jasmine.createSpy('onSubmit'),
        onCancel: jasmine.createSpy('onCancel'),
    };

    beforeEach(() => {
        defaultProps.onChange.calls.reset();
        defaultProps.onSubmit.calls.reset();
        defaultProps.onCancel.calls.reset();
    });

    it('should call onChange when name is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: 'foo' } };
        wrapper.find('input.name').simulate('change', evt);
        expect(defaultProps.onChange).toHaveBeenCalledWith('name', evt.target.value);
    });

    it('should call onChange when description is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: 'foo' } };
        wrapper.find('textarea.description').simulate('change', evt);
        expect(defaultProps.onChange).toHaveBeenCalledWith('description', evt.target.value);
    });

    it('should call onChange when duration is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        wrapper.find('DurationInput').prop('onChange')(44);
        expect(defaultProps.onChange).toHaveBeenCalledWith('duration', 44);
    });

    it('should call onChange when interval is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: 1 } };
        wrapper.find('input.interval').simulate('change', evt);
        expect(defaultProps.onChange).toHaveBeenCalledWith('interval', evt.target.value);
    });

    it('should call onSubmit when form is submitted', () => {
        const preventDefault = jasmine.createSpy('preventDefault');
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        wrapper.find('form').simulate('submit', { preventDefault });
        expect(defaultProps.onSubmit).toHaveBeenCalled();
    });

    it('should call onCancel when cancel button is clicked', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        wrapper.find('button.cancel').simulate('click');
        expect(defaultProps.onCancel).toHaveBeenCalled();
    });
});
