import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobRemoveDialogue from
    '../../src/components/ScanningJobRemoveDialogue';

describe('<ScanningJobRemoveDialogue/>', () => {
    const defaultProps = {
        onCancel: () => {},
        onConfirm: () => {},
        name: 'My scan job',
    };

    it('should render "Remove Job?"', () => {
        const wrapper = shallow(<ScanningJobRemoveDialogue {...defaultProps} />);
        expect(wrapper.text()).toContain('Remove Job?');
    });

    it('should render a warning', () => {
        const wrapper = shallow(<ScanningJobRemoveDialogue
            {...defaultProps}
            name="My Job"
        />);
        expect(wrapper.text())
            .toContain('This will permanently remove the planned job My Job with no ability to undo this action.');
    });

    it('should render a Yes button', () => {
        const wrapper = shallow(<ScanningJobRemoveDialogue
            {...defaultProps}
        />);
        const btn = wrapper.find('.confirm-button');
        expect(btn.exists()).toBeTruthy();
        expect(btn.text()).toEqual('Yes');
        expect(btn.prop('className')).toContain('btn-primary');
    });

    it('should render a No button', () => {
        const wrapper = shallow(<ScanningJobRemoveDialogue
            {...defaultProps}
        />);
        const btn = wrapper.find('.cancel-button');
        expect(btn.exists()).toBeTruthy();
        expect(btn.text()).toEqual('No');
    });

    it('should call onConfirm when clicking Yes', () => {
        const onConfirm = jasmine.createSpy('onConfirm');
        const wrapper = shallow(<ScanningJobRemoveDialogue
            {...defaultProps}
            onConfirm={onConfirm}
        />);
        const btn = wrapper.find('.confirm-button');
        btn.simulate('click');
        expect(onConfirm).toHaveBeenCalled();
    });

    it('should call onCancel when clicking No', () => {
        const onCancel = jasmine.createSpy('onCancel');
        const wrapper = shallow(<ScanningJobRemoveDialogue
            {...defaultProps}
            onCancel={onCancel}
        />);
        const btn = wrapper.find('.cancel-button');
        btn.simulate('click');
        expect(onCancel).toHaveBeenCalled();
    });
});

