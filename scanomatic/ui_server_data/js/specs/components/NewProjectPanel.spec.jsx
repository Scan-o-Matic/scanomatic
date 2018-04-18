import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import NewProjectPanel from '../../src/components/NewProjectPanel';

describe('<NewProjectPanel/>', () => {
    const defaultProps = {
        name: '', description: '',
    };

    it('should call onChange when name is changed', () => {
        const onChange = jasmine.createSpy('onChange');
        const wrapper = shallow(<NewProjectPanel {...defaultProps} onChange={onChange} />);
        wrapper.find('input.name').simulate('change', { target: { value: 'foo' } });
        expect(onChange).toHaveBeenCalledWith('name', 'foo');
    });

    it('should call onChange when description is changed', () => {
        const onChange = jasmine.createSpy('onChange');
        const wrapper = shallow(<NewProjectPanel {...defaultProps} onChange={onChange} />);
        wrapper.find('textarea.description').simulate('change', { target: { value: 'foo' } });
        expect(onChange).toHaveBeenCalledWith('description', 'foo');
    });

    it('should call onSubmit when form is submitted', () => {
        const onSubmit = jasmine.createSpy('onSubmit');
        const preventDefault = jasmine.createSpy('preventDefault');
        const wrapper = shallow(<NewProjectPanel {...defaultProps} onSubmit={onSubmit} />);
        wrapper.find('form').simulate('submit', { preventDefault });
        expect(onSubmit).toHaveBeenCalled();
    });

    it('should call onCancel when cancel button is clicked', () => {
        const onCancel = jasmine.createSpy('onCancel');
        const wrapper = shallow(<NewProjectPanel {...defaultProps} onCancel={onCancel} />);
        wrapper.find('button.cancel').simulate('click');
        expect(onCancel).toHaveBeenCalled();
    });
});
