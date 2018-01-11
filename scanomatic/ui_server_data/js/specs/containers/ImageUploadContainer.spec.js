import { mount } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ImageUploadContainer from '../../src/containers/ImageUploadContainer';
import * as helpers from '../../src/helpers';
import cccMetadata from '../fixtures/cccMetadata';
import FakePromise from '../helpers/FakePromise';


describe('<ImageUploadContainer />', () => {
    const fixture = 'MyFixture';
    const cccId = 'CCC0';
    const token = 'T0K3N';
    const imageId = 'IMG0';
    const onFinish = jasmine.createSpy('onFinish');
    const props = { cccMetadata, onFinish };
    let uploadPromise;

    beforeEach(() => {
        onFinish.calls.reset();
        uploadPromise = new FakePromise();
        spyOn(helpers, 'uploadImage').and.callFake((_1, _2, _3, _4, progress) => {
            if (progress) progress(1, 2, 'Fake uploading image');
            return uploadPromise;
        });
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
        helpers.uploadImage.and.returnValue(new Promise(() => {}));
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
        expect(helpers.uploadImage).toHaveBeenCalledWith(
            cccMetadata.id, image, cccMetadata.fixtureName,
            cccMetadata.accessToken, jasmine.any(Function),
        );
    });

    it('should pass the progress to the children', () => {
        const image = new File(['foo'], 'myimage.tiff');
        const wrapper = mount(<ImageUploadContainer {...props} />);
        wrapper.find('ImageUpload').prop('onImageChange')(image);
        const setProgress = helpers.uploadImage.calls.argsFor(0)[4];
        setProgress(3, 4, 'You are here');
        wrapper.update();
        expect(wrapper.children().prop('progress'))
            .toEqual({ now: 3, max: 4, text: 'You are here' });
    });

    describe('when upload succeed', () => {
        const image = new File(['foo'], 'myimage.tiff');

        beforeEach(() => {
            uploadPromise.value = imageId;
        });

        it('should clear the image', () => {
            const wrapper = mount(<ImageUploadContainer {...props} />);
            wrapper.find('ImageUpload').prop('onImageChange')(image);
            wrapper.update();
            expect(wrapper.children().prop('image')).toBe(null);
        });

        it('should call onFinish with the image name and id', () => {
            const wrapper = mount(<ImageUploadContainer {...props} />);
            wrapper.find('ImageUpload').prop('onImageChange')(image);
            expect(onFinish)
                .toHaveBeenCalledWith({ name: 'myimage.tiff', id: imageId });
        });

        it('should clear the progress on upload success', () => {
            const wrapper = mount(<ImageUploadContainer {...props} />);
            wrapper.find('ImageUpload').prop('onImageChange')(image);
            wrapper.update();
            expect(wrapper.children().prop('progress')).toBeFalsy();
        });
    });

    describe('when upload fails', () => {
        const image = new File(['foo'], 'myimage.tiff');

        beforeEach(() => {
            uploadPromise.error = 'XxX';
        });

        it('should show an alert', () => {
            const wrapper = mount(<ImageUploadContainer {...props} />);
            wrapper.find('ImageUpload').prop('onImageChange')(image);
            expect(window.alert)
                .toHaveBeenCalledWith('An error occured while uploading the image: XxX');
        });

        it('should clear the image', () => {
            const wrapper = mount(<ImageUploadContainer {...props} />);
            wrapper.find('ImageUpload').prop('onImageChange')(image);
            wrapper.update();
            expect(wrapper.children().prop('image')).toBe(null);
        });

        it('should clear the progress', () => {
            const wrapper = mount(<ImageUploadContainer {...props} />);
            wrapper.find('ImageUpload').prop('onImageChange')(image);
            wrapper.update();
            expect(wrapper.children().prop('progress')).toBeFalsy();
        });
    });
});
