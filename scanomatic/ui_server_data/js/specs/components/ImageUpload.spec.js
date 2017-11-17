import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ImageUpload from '../../ccc/components/ImageUpload';

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
});
