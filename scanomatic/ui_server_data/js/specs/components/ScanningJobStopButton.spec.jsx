import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import ScanningJobStopButton from
    '../../src/components/ScanningJobStopButton';

describe('<ScanningJobStopButton/>', () => {
    const props = {
        onStopJob: () => {},
    };

    it('should render a button', () => {
        const wrapper = shallow(<ScanningJobStopButton {...props} />);
        expect(wrapper.find('button').exists()).toBeTruthy();
    });

    it('should render a remove icon', () => {
        const wrapper = shallow(<ScanningJobStopButton {...props} />);
        expect(wrapper
            .find('button')
            .find('span.glyphicon.glyphicon-remove')
            .exists()).toBeTruthy();
    });

    it('should render "Stop"', () => {
        const wrapper = shallow(<ScanningJobStopButton {...props} />);
        expect(wrapper.text()).toContain('Stop');
    });

    it('should call onStopJob when clicked', () => {
        const onStopJob = jasmine.createSpy('onStopJob');
        const wrapper = shallow(<ScanningJobStopButton
            {...props}
            onStopJob={onStopJob}
        />);
        wrapper.simulate('click');
        expect(onStopJob).toHaveBeenCalled();
    });
});
