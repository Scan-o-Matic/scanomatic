import { mount } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ImageUploadContainer from '../../ccc/containers/ImageUploadContainer';
import * as helpers from '../../ccc/helpers';


describe('<ImageUploadContainer />', () => {
    const fixture = 'MyFixture';
    const cccId = 'CCC0';
    const token = 'T0K3N';
    const imageId = 'IMG0';
    const onFinish = jasmine.createSpy('onFinish');
    const props = { cccId, fixture, token, onFinish };

    beforeEach(() => {
        onFinish.calls.reset();
        spyOn(helpers, 'uploadImage').and.returnValue({ then: f => {
            f(imageId);
            return { catch: () => {} };
        }});
        spyOn(window, 'alert');
    });

    it('should render an <ImageUpload />', () => {
        const wrapper = mount(<ImageUploadContainer {...props} />);
        expect(wrapper.find('ImageUpload').exists()).toBeTruthy();
    });

    it('should initially pass no image', () => {
        const wrapper = mount(<ImageUploadContainer {...props} />);
        expect(wrapper.find('ImageUpload').prop('image')).toBe(null);
    });

    it('should update the image when <ImageUpload /> calls onImageChange', () => {
        const image = new File(['foo'], 'myimage.tiff');
        const wrapper = mount(<ImageUploadContainer {...props} />);
        wrapper.find('ImageUpload').prop('onImageChange')(image);
        wrapper.update();
        expect(wrapper.find('ImageUpload').prop('image')).toEqual(image);
    });

    it('should upload the image when <ImageUpload /> calls onImageChange', () => {
        const image = new File(['foo'], 'myimage.tiff');
        const wrapper = mount(<ImageUploadContainer {...props} />);
        wrapper.find('ImageUpload').prop('onImageChange')(image);
        expect(helpers.uploadImage).toHaveBeenCalledWith('CCC0', image, fixture, token);
    });

    it('should call onFinish with the image name and id when upload succeed', () => {
        const image = new File(['foo'], 'myimage.tiff');
        const wrapper = mount(<ImageUploadContainer {...props} />);
        wrapper.find('ImageUpload').prop('onImageChange')(image);
        expect(onFinish)
            .toHaveBeenCalledWith({ name: 'myimage.tiff', id: imageId });
    });

    it('should show an alert if the upload fails', () => {
        helpers.uploadImage.and
            .returnValue({ then: () => ({ catch: f => f('XxX') }) });
        const image = new File(['foo'], 'myimage.tiff');
        const wrapper = mount(<ImageUploadContainer {...props} />);
        wrapper.find('ImageUpload').prop('onImageChange')(image);
        expect(window.alert)
            .toHaveBeenCalledWith('An error occured while uploading the image: XxX');
    });
});
