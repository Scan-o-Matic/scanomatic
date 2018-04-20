import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import NewExperimentPanel from './NewExperimentPanel';
import Duration from '../Duration';

describe('<NewExperimentPanel/>', () => {
    const defaultProps = {
        project: 'I project',
        name: 'Cool idea',
        description: 'Test all the things',
        duration: new Duration(),
        interval: new Duration(),
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
        onNameChange: jasmine.createSpy('onNameChange'),
        onDescriptionChange: jasmine.createSpy('onDescriptionChange'),
        onDurationDaysChange: jasmine.createSpy('onDurationDaysChange'),
        onDurationHoursChange: jasmine.createSpy('onDurationHoursChange'),
        onDurationMinutesChange: jasmine.createSpy('onDurationMinutesChange'),
        onIntervalChange: jasmine.createSpy('onIntervalChange'),
        onScannerChange: jasmine.createSpy('onIntervalChange'),
        onSubmit: jasmine.createSpy('onSubmit'),
        onCancel: jasmine.createSpy('onCancel'),
    };

    beforeEach(() => {
        defaultProps.onNameChange.calls.reset();
        defaultProps.onDescriptionChange.calls.reset();
        defaultProps.onDurationDaysChange.calls.reset();
        defaultProps.onDurationHoursChange.calls.reset();
        defaultProps.onDurationMinutesChange.calls.reset();
        defaultProps.onIntervalChange.calls.reset();
        defaultProps.onScannerChange.calls.reset();
        defaultProps.onSubmit.calls.reset();
        defaultProps.onCancel.calls.reset();
    });

    it('should call onChangeName when name is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: 'foo' } };
        wrapper.find('input.name').simulate('change', evt);
        expect(defaultProps.onNameChange).toHaveBeenCalledWith(evt.target.value);
    });

    it('should call onDescriptionChange when description is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: 'foo' } };
        wrapper.find('textarea.description').simulate('change', evt);
        expect(defaultProps.onDescriptionChange).toHaveBeenCalledWith(evt.target.value);
    });

    it('should call onDurationDaysChange when duration days is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: '1' } };
        wrapper.find('input.days').simulate('change', evt);
        expect(defaultProps.onDurationDaysChange).toHaveBeenCalledWith(evt.target.value);
    });

    it('should call onDurationHoursChange when duration hours is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: '1' } };
        wrapper.find('input.hours').simulate('change', evt);
        expect(defaultProps.onDurationHoursChange).toHaveBeenCalledWith(evt.target.value);
    });

    it('should call onDurationMinutesChange when duration minutes is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: '1' } };
        wrapper.find('input.minutes').simulate('change', evt);
        expect(defaultProps.onDurationMinutesChange).toHaveBeenCalledWith(evt.target.value);
    });

    it('should call onIntervalChange when interval is changed', () => {
        const wrapper = shallow(<NewExperimentPanel {...defaultProps} />);
        const evt = { target: { value: '1' } };
        wrapper.find('input.interval').simulate('change', evt);
        expect(defaultProps.onIntervalChange).toHaveBeenCalledWith(evt.target.value);
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
