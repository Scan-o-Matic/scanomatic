import { shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import CCCEditor from '../../ccc/components/CCCEditor';

describe('<CCCEditor />', () => {
    const onFinishPlate = jasmine.createSpy('onFinishPlate');
    const onFinishUpload = jasmine.createSpy('onFinishUpload');
    const props = {
        pinFormat: [2, 3],
        accessToken: 'T0P53CR3T',
        cccId: 'CCC42',
        fixtureName: 'MyFixture',
        plates: [
            { imageName: 'my-image.tiff', imageId: '1M4G3', plateId: 1 },
            { imageName: 'my-image.tiff', imageId: '1M4G3', plateId: 2 },
            { imageName: 'other-image.tiff', imageId: '1M4G32', plateId: 2 },
        ],
        currentPlate: 1,
        onFinishPlate,
        onFinishUpload,
    };

    it('should render an <PolynomialConstructionContainer/>', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PolynomialConstructionContainer').exists())
            .toBeTruthy();
    });

    it('should pass cccId to <PolynomialConstructionContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PolynomialConstructionContainer').prop('cccId'))
            .toEqual(props.cccId);
    });

    it('should pass accessToken to <PolynomialConstructionContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PolynomialConstructionContainer')
            .prop('accessToken')).toEqual(props.accessToken);
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

    it('should render a <PlateEditorContainer /> per plate', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').length).toEqual(3);
    });

    it('should pass cccId to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('cccId'))
            .toEqual(props.cccId);
    });

    it('should pass accessToken to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('accessToken'))
            .toEqual(props.accessToken);
    });

    it('should pass pinFormat to <PlateEditorContainer />', () => {
        const wrapper = shallow(<CCCEditor {...props} />);
        expect(wrapper.find('PlateEditorContainer').at(0).prop('pinFormat'))
            .toEqual(props.pinFormat);
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
