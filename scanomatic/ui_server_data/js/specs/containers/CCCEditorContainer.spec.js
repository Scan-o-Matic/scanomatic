import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import CCCEditorContainer from '../../ccc/containers/CCCEditorContainer';


describe('<CCCEditorContainer />', () => {
    const props = {
        pinFormat: [2, 3],
        accessToken: 'T0P53CR3T',
        cccId: 'CCC42',
        fixtureName: 'MyFixture',
    };

    const image = {
        name: 'new-image.tiff',
        id: 'NewImg0',
    };

    it('should render a <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').exists()).toBeTruthy();
    });

    it('should pass cccId to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('cccId')).toEqual(props.cccId);
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

    it('should pass an empty image list to <CCCEditor />', () => {
        const wrapper = shallow(<CCCEditorContainer {...props} />);
        expect(wrapper.find('CCCEditor').prop('images')).toEqual([]);
    });

    describe('when CCCEditor calls onFinishUpload', () => {
        it('should add the new image to the list', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.find('CCCEditor').prop('onFinishUpload')(image);
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('images')).toEqual([image]);
        });

        it('should set the current image', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.find('CCCEditor').prop('onFinishUpload')(image);
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('currentImage')).toEqual(0);
        });
    });

    describe('when <CCCEditor /> calls onFinishImage', () => {
        it('should clear the current image', () => {
            const wrapper = shallow(<CCCEditorContainer {...props} />);
            wrapper.setState({ currentImage: 123 });
            wrapper.find('CCCEditor').prop('onFinishImage')();
            wrapper.update();
            expect(wrapper.find('CCCEditor').prop('currentImage')).toBe(null);
        });
    });
});
