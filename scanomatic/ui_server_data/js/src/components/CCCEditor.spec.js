import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import CCCEditor from '../../src/components/CCCEditor';
import cccMetadata from '../fixtures/cccMetadata';

describe('<CCCEditor />', () => {
    const onFinalizeCCC = jasmine.createSpy('onFinalizeCCC');
    const onFinishPlate = jasmine.createSpy('onFinishPlate');
    const onFinishUpload = jasmine.createSpy('onFinishUpload');
    const props = {
        cccMetadata,
        plates: [
            { imageName: 'my-image.tiff', imageId: '1M4G3', plateId: 1 },
            { imageName: 'my-image.tiff', imageId: '1M4G3', plateId: 2 },
            { imageName: 'other-image.tiff', imageId: '1M4G32', plateId: 2 },
        ],
        currentPlate: 1,
        onFinalizeCCC,
        onFinishPlate,
        onFinishUpload,
    };

    it('should render a <CCCInfoBox />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('CCCInfoBox').exists()).toBeTruthy();
    });

    it('should pass cccMetadata to <CCCInfoBox />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('CCCInfoBox').prop('cccMetadata')).toEqual(cccMetadata);
    });

    it('should render an <PolynomialConstructionContainer/>', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PolynomialConstructionContainer').exists())
            .toBeTruthy();
    });

    it('should pass cccMetadata to <PolynomialConstructionContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PolynomialConstructionContainer').prop('cccMetadata'))
            .toEqual(cccMetadata);
    });

    it('should pass onFinalizeCCC to <PolynomialConstructionContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PolynomialConstructionContainer').prop('onFinalizeCCC'))
            .toBe(onFinalizeCCC);
    });

    describe('when currentImage is null', () => {
        it('should render an <ImageUploadContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('ImageUploadContainer').exists()).toBeTruthy();
        });

        it('should pass cccMetadata to <ImageUploadContainer />', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            expect(wrapper.find('ImageUploadContainer').prop('cccMetadata'))
                .toEqual(cccMetadata);
        });

        it('should call onFinishUpload when <ImageUploadContainer /> calls onFinish', () => {
            const wrapper = shallow(<CCCEditor {...props} />);
            wrapper.find('ImageUploadContainer').prop('onFinish')();
            expect(onFinishUpload).toHaveBeenCalled();
        });
    });

    it('should render a <PlateEditorContainer /> per plate', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').length).toEqual(3);
    });

    it('should pass cccMetadata to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('cccMetadata'))
            .toEqual(cccMetadata);
    });

    it('should pass the plate imageId to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('imageId'))
            .toEqual(props.plates[0].imageId);
    });

    it('should pass the plate imageName to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('imageName'))
            .toEqual(props.plates[0].imageName);
    });

    it('should pass the plate plateId to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('plateId'))
            .toEqual(props.plates[0].plateId);
    });

    it('should pass collapse=false to <PlateEditorContainer /> for current plate', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(1).prop('collapse'))
            .toBeFalsy();
    });

    it('should pass collapse=true to <PlateEditorContainer /> otherwise', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('collapse'))
            .toBeTruthy();
        expect(wrapper.find('PlateEditorContainer').at(2).prop('collapse'))
            .toBeTruthy();
    });

    it('should call onFinishPlate when <PlateEditorContainer /> calls onFinish', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        wrapper.find('PlateEditorContainer').at(1).prop('onFinish')();
        expect(onFinishPlate).toHaveBeenCalled();
    });
});
