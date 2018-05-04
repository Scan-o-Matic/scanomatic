import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobRemoveButton from
    '../../src/components/ScanningJobRemoveButton';

describe('<ScanningJobRemoveButton/>', () => {
    const props = {
        onRemoveJob: () => {},
    };

    it('Renders a button', () => {
        const wrapper = shallow(<ScanningJobRemoveButton {...props} />);
        expect(wrapper.find('button').exists()).toBeTruthy();
    });

    it('Renders a remove icon', () => {
        const wrapper = shallow(<ScanningJobRemoveButton {...props} />);
        expect(wrapper
            .find('button')
            .find('span.glyphicon.glyphicon-remove')
            .exists()).toBeTruthy();
    });

    it('should render "Remove"', () => {
        const wrapper = shallow(<ScanningJobRemoveButton {...props} />);
        expect(wrapper.text()).toContain('Remove');
    });

    it('should call onRemoveJob when clicked', () => {
        const onRemoveJob = jasmine.createSpy('onRemoveJob');
        const wrapper = shallow(<ScanningJobRemoveButton
            {...props}
            onRemoveJob={onRemoveJob}
        />);
        wrapper.simulate('click');
        expect(onRemoveJob).toHaveBeenCalled();
    });
});
