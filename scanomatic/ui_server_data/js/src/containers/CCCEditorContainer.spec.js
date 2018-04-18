import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import CCCEditorContainer from '../../src/containers/CCCEditorContainer';
import * as API from '../../src/api';
import cccMetadata from '../fixtures/cccMetadata';
import FakePromise from '../helpers/FakePromise';

describe('<CCCEditorContainer />', () => {
    const props = {
        cccMetadata, onFinalizeCCC: jasmine.createSpy('onFinalizeCCC'),
    };

    const image = {
        name: 'new-image.tiff',
        id: 'NewImg0',
    };

    beforeEach(() => {
        spyOn(API, 'GetFixturePlates').and.returnValue(new FakePromise());
    });

    it('should render a <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').exists()).toBeTruthy();
    });

    it('should pass cccMetadata to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('cccMetadata')).toEqual(props.cccMetadata);
    });

    it('should pass onFinalizeCCC to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('onFinalizeCCC'))
            .toBe(props.onFinalizeCCC);
    });

    it('should pass accessToken to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('accessToken')).toEqual(props.accessToken);
    });

    it('should pass pinFormat to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('pinFormat')).toEqual(props.pinFormat);
    });

    it('should pass fixtureName to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('fixtureName')).toEqual(props.fixtureName);
    });

    it('should pass an empty plate list to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('plates')).toEqual([]);
    });

    it('should initialy pass ready=false to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('ready')).toBeFalsy();
    });

    it('should load the number of plates from the API', () => {
        shallow(<CCCEditorContainer {...props} />);
        expect(API.GetFixturePlates).toHaveBeenCalledWith('MyFixture');
    });

    it('should pass ready=true to <CCCEditor /> when GetFixturePlates resolves', () => {
        API.GetFixturePlates.and.returnValue(FakePromise.resolve([{}, {}, {}]));
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        wrapper.update();
        expect(wrapper.find('CCCEditor').prop('ready')).toBeTruthy();
    });

    describe('when CCCEditor calls onFinishUpload', () => {
        const oldPlate = {imageId: 'OldImg0', imageName: 'foo.tiff', plateId: 1 };

        beforeEach(() => {
            API.GetFixturePlates.and.returnValue(FakePromise.resolve([{}, {}, {}]));
        });

        it('should add new plates to the list', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.setState({ plates: [oldPlate] });
            wrapper.find('CCCEditor').prop('onFinishUpload')(image);
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('plates')).toEqual([
                oldPlate,
                {imageId: image.id, imageName: image.name, plateId: 1},
                {imageId: image.id, imageName: image.name, plateId: 2},
                {imageId: image.id, imageName: image.name, plateId: 3},
            ]);
        });

        it('should set the current plate if null', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.setState({ plates: [oldPlate], currentPlate: null });
            wrapper.find('CCCEditor').prop('onFinishUpload')(image);
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('currentPlate')).toEqual(1);
        });

        it('should not change the current plate if set', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.setState({ plates: [oldPlate], currentPlate: 0 });
            wrapper.find('CCCEditor').prop('onFinishUpload')(image);
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('currentPlate')).toEqual(0);
        });
    });

    describe('when <CCCEditor /> calls onFinishPlate', () => {
        it('should move to the next plate if any', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.setState({
                currentPlate: 1,
                plates: [
                    {imageId: image.id, imageName: image.name, plateId: 1},
                    {imageId: image.id, imageName: image.name, plateId: 2},
                    {imageId: image.id, imageName: image.name, plateId: 3},
                ],
            });
            wrapper.find('CCCEditor').prop('onFinishPlate')();
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('currentPlate')).toEqual(2);
        });

        it('should clear the current plate if last', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.setState({
                currentPlate: 2,
                plates: [
                    {imageId: image.id, imageName: image.name, plateId: 1},
                    {imageId: image.id, imageName: image.name, plateId: 2},
                    {imageId: image.id, imageName: image.name, plateId: 3},
                ],
            });
            wrapper.find('CCCEditor').prop('onFinishPlate')();
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('currentPlate')).toEqual(null);
        });
    });
});
