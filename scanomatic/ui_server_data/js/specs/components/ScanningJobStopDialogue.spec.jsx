import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobStopDialogue from
    '../../src/components/ScanningJobStopDialogue';

describe('<ScanningJobStopDialogue/>', () => {
    const defaultProps = {
        onCancel: () => {},
        onConfirm: () => {},
        name: 'My scan job',
    };

    it('should call onConfirm with the given reason when clicking Yes', () => {
        const onConfirm = jasmine.createSpy('onConfirm');
        const wrapper = shallow(<ScanningJobStopDialogue
            {...defaultProps}
            onConfirm={onConfirm}
        />);
        const input = wrapper.find('.reason-input');
        input.simulate('change', { target: { value: 'The Reason' } });
        const btn = wrapper.find('.confirm-button');
        btn.simulate('click');
        expect(onConfirm).toHaveBeenCalledWith('The Reason');
    });

    it('should call onCancel when clicking No', () => {
        const onCancel = jasmine.createSpy('onCancel');
        const wrapper = shallow(<ScanningJobStopDialogue
            {...defaultProps}
            onCancel={onCancel}
        />);
        const btn = wrapper.find('.cancel-button');
        btn.simulate('click');
        expect(onCancel).toHaveBeenCalled();
    });
});

