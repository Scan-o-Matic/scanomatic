import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ImageUpload from '../../src/components/ImageUpload';

describe('<ImageUpload />', () => {
    const onImageChange = jasmine.createSpy('onFileChange');
    const props = { onImageChange };

    it('should render a file <input />', () => {
        const wrapper = shallow(<ImageUpload {...props} />);
        expect(wrapper.find('input[type="file"]').exists()).toBeTruthy();
    });

    it('should call onFileChange when a file is selected', () => {
        const file = 'my-image.tiff';
        const wrapper = shallow(<ImageUpload {...props} />);
        wrapper.find('input[type="file"]')
            .simulate('change', { target: { files: [file] } });
        expect(onImageChange).toHaveBeenCalledWith(file);
    });

    const progress = { now: 1, max: 2, text: 'Making progress' };

    it('should hide the file input if progress is not null', () => {
        const wrapper = shallow(<ImageUpload {...props} progress={progress} />);
        expect(wrapper.find('input[type="file"]').exists()).toBeFalsy();
    });

    it('should show the progress text', () => {
        const wrapper = shallow(<ImageUpload {...props} progress={progress} />);
        expect(wrapper.text()).toContain('Making progress');
    });

    it('should show a progress bar if progress is not null', () => {
        const wrapper = shallow(<ImageUpload {...props} progress={progress} />);
        const progressBar = wrapper.find('.progress-bar');
        expect(progressBar.exists()).toBeTruthy();
        expect(progressBar.prop('style').width).toEqual('50%');
    });
});
