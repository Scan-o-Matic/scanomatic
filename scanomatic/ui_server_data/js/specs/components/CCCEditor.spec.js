import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import CCCEditor from '../../ccc/components/CCCEditor';

describe('<CCCEditor />', () => {
    const onFinishImage = jasmine.createSpy('onFinishImage');
    const onFinishUpload = jasmine.createSpy('onFinishUpload');
    const props = {
        pinFormat: [2, 3],
        accessToken: 'T0P53CR3T',
        cccId: 'CCC42',
        fixtureName: 'MyFixture',
        images: [
            { imageName: 'my-image.tiff', imageId: '1M4G3' },
            { imageName: 'my-image.tiff', imageId: '1M4G3' },
            { imageName: 'other-image.tiff', imageId: '1M4G32' },
        ],
        onFinishImage,
        onFinishUpload,
    };

    it('should render an <PlateList />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateList').exists()).toBeTruthy();
    });

    it('should pass images to <PlateList />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateList').prop('plates')).toEqual(props.images);
    });

    describe('when currentImage is null', () => {
        it('should render an <ImageUploadContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('ImageUploadContainer').exists()).toBeTruthy();
        });

        it('should pass cccId to <ImageUploadContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('ImageUploadContainer').prop('cccId'))
                .toEqual(props.cccId);
        });

        it('should pass accessToken to <ImageUploadContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('ImageUploadContainer').prop('token'))
                .toEqual(props.accessToken);
        });

        it('should pass fixtureName to <ImageUploadContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('ImageUploadContainer').prop('fixture'))
                .toEqual(props.fixtureName);
        });

        it('should call onFinishUpload when <ImageUploadContainer /> calls onFinish', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            wrapper.find('ImageUploadContainer').prop('onFinish')();
            expect(onFinishUpload).toHaveBeenCalled();
        });
    });

    describe('when currentImage is set', () => {
        beforeEach(() => {
            props.currentImage = 1;
        });

        it('should render a <PlateEditorContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('PlateEditorContainer').exists()).toBeTruthy();
        });

        it('should pass cccId to <PlateEditorContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('PlateEditorContainer').prop('cccId'))
                .toEqual(props.cccId);
        });

        it('should pass accessToken to <PlateEditorContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('PlateEditorContainer').prop('accessToken'))
                .toEqual(props.accessToken);
        });

        it('should pass pinFormat to <PlateEditorContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('PlateEditorContainer').prop('pinFormat'))
                .toEqual(props.pinFormat);
        });

        it('should pass current imageId to <PlateEditorContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('PlateEditorContainer').prop('imageId'))
                .toEqual(props.images[1].id);
        });

        it('should pass current plateId to <PlateEditorContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('PlateEditorContainer').prop('plateId')).toEqual(1);
        });

        it('should call onFinishPlate when <PlateEditorContainer /> calls onFinish', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            wrapper.find('PlateEditorContainer').prop('onFinish')();
            expect(onFinishImage).toHaveBeenCalled();
        });
    });
});
