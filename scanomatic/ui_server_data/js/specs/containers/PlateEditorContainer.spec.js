import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import PlateEditorContainer from '../../ccc/containers/PlateEditorContainer';
import * as API from '../../ccc/api';

describe('<PlateEditorContainer />', () => {
    const props = {
        pinFormat: [2, 3],
        accessToken: 'T0P53CR3T',
        cccId: 'CCC42',
        imageId: '1M4G3',
        plateId: 'PL4T3',
        onFinish: jasmine.createSpy('onFinish'),
    };

    beforeEach(() => {
        props.onFinish.calls.reset();
        spyOn(API, 'SetGrayScaleTransform').and.returnValue(Promise.resolve({}));
    });

    it('should render a <PlateEditor />', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        expect(wrapper.find('PlateEditor').exists()).toBeTruthy();
    });

    it('should start with the transforming step', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        expect(wrapper.prop('step')).toEqual('transforming');
    });

    it('should call SetGrayScaleTransform', () => {
        shallow(<PlateEditorContainer {...props} />);
        expect(API.SetGrayScaleTransform)
            .toHaveBeenCalledWith(props.cccId, props.imageId, props.plateId, props.accessToken)
    });

    it('should switch to the gridding step when SetGrayScaleTransform finishes', (done) => {
        const promise = Promise.resolve({});
        API.SetGrayScaleTransform.and.returnValue(promise);
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        promise.then(() => {
            wrapper.update();
            expect(wrapper.prop('step')).toEqual('gridding');
            done();
        });
    });

    it('should switch to the colony step when onGriddingFinish is called', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        wrapper.prop('onGriddingFinish')();
        wrapper.update();
        expect(wrapper.prop('step')).toEqual('colony');
    });

    it('should start the colony step with colony at position 0 0', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        wrapper.prop('onGriddingFinish')();
        wrapper.update();
        expect(wrapper.prop('selectedColony')).toEqual({ row: 0, col: 0 });
    });

    it('should move to the next colony when onColonyFinish is called', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        wrapper.prop('onGriddingFinish')();
        wrapper.prop('onColonyFinish')();
        wrapper.update();
        expect(wrapper.prop('selectedColony')).toEqual({ row: 0, col: 1 });
    });

    it('should move to the next row when necessary', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        wrapper.prop('onGriddingFinish')();
        wrapper.prop('onColonyFinish')();
        wrapper.prop('onColonyFinish')();
        wrapper.update();
        expect(wrapper.prop('selectedColony')).toEqual({ row: 1, col: 0 });
    });

    it('should call the onFinish callback after the last colony', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        wrapper.prop('onGriddingFinish')();
        wrapper.setState({ selectedColony: { row: 2, col: 1 } });
        wrapper.prop('onColonyFinish')();
        wrapper.update();
        expect(props.onFinish).toHaveBeenCalled();
    });
});
