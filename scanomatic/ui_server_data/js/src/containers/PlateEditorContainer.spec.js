import React from 'react';
import { shallow } from 'enzyme';

import '../components/enzyme-setup';
import PlateEditorContainer from '../../src/containers/PlateEditorContainer';
import * as API from '../../src/api';
import cccMetadata from '../fixtures/cccMetadata';

describe('<PlateEditorContainer />', () => {
    const props = {
        cccMetadata,
        imageId: '1M4G3',
        imageName: 'myimage.tiff',
        onFinish: jasmine.createSpy('onFinish'),
        plateId: 1,
    };

    beforeEach(() => {
        props.onFinish.calls.reset();
        spyOn(API, 'SetGrayScaleTransform')
            .and.returnValue(new Promise(() => {}));
        spyOn(API, 'SetGridding')
            .and.returnValue(new Promise(() => {}));
    });

    it('should render a <PlateEditor />', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        expect(wrapper.find('PlateEditor').exists()).toBeTruthy();
    });

    it('should pass cccMetadata to <PlateEditor />', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        expect(wrapper.find('PlateEditor').prop('cccMetadata')).toEqual(cccMetadata);
    });

    it('should start with the pre-processing step', () => {
        const wrapper = shallow(<PlateEditorContainer {...props} />);
        expect(wrapper.prop('step')).toEqual('pre-processing');
    });

    describe('pre-processing', () => {
        it('should call SetGrayScaleTransform', () => {
            shallow(<PlateEditorContainer {...props} />);
            expect(API.SetGrayScaleTransform)
                .toHaveBeenCalledWith(
                    props.cccMetadata.id, props.imageId, props.plateId,
                    props.cccMetadata.accessToken,
                );
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
    });

    describe('gridding', () => {
        beforeEach(() => {
            API.SetGrayScaleTransform.and.returnValue({ then: f => f() });
        });

        it('should call SetGridding with offset 0,0', () => {
            shallow(<PlateEditorContainer {...props} />);
            expect(API.SetGridding).toHaveBeenCalledWith(
                props.cccMetadata.id, props.imageId, props.plateId,
                [props.cccMetadata.pinningFormat.nCols, props.cccMetadata.pinningFormat.nRows],
                [0, 0], props.cccMetadata.accessToken,
            );
        });

        it('should set griddingLoading to true', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.update();
            expect(wrapper.prop('griddingLoading')).toBeTruthy();
        });

        it('should get a new grid using the offsets when onRegrid is called', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            API.SetGridding.calls.reset();
            wrapper.setState( { rowOffset: 2, colOffset: 3 });
            wrapper.prop('onRegrid')();
            expect(API.SetGridding).toHaveBeenCalledWith(
                props.cccMetadata.id, props.imageId, props.plateId,
                [props.cccMetadata.pinningFormat.nCols, props.cccMetadata.pinningFormat.nRows],
                [2, 3], props.cccMetadata.accessToken,
            );
        });

        it('should update rowOffset when onRowOffsetChange is called', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.prop('onRowOffsetChange')(4);
            wrapper.update();
            expect(wrapper.prop('rowOffset')).toEqual(4);
        });

        it('should update colOffset when onRowOffsetChange is called', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.prop('onColOffsetChange')(5);
            wrapper.update();
            expect(wrapper.prop('colOffset')).toEqual(5);
        });

        describe('on gridding error', () => {
            const errorData = { reason: 'bad', grid: [[[0]], [[0]]] };
            beforeEach(() => {
                API.SetGridding.and.returnValue({ then: (f, g) => g(errorData) });
            });

            it('should set griddingLoading to false', () => {
                const wrapper = shallow(<PlateEditorContainer {...props} />);
                wrapper.update();
                expect(wrapper.prop('griddingLoading')).toBeFalsy();
            });

            it('should pass the gridding error', () => {
                const wrapper = shallow(<PlateEditorContainer {...props} />);
                wrapper.update();
                expect(wrapper.prop('griddingError')).toEqual('bad');
            });

            it('sholud pass the grid to the component', () => {
                const wrapper = shallow(<PlateEditorContainer {...props} />);
                expect(wrapper.prop('grid')).toEqual(errorData.grid);

            });

        });

        describe('on gridding success', () => {
            const grid = [[[0]], [[0]]];

            beforeEach(() => {
                API.SetGridding.and.returnValue({ then: (f) => f({ grid }) });
            });


            it('should set griddingLoading to false', () => {
                const wrapper = shallow(<PlateEditorContainer {...props} />);
                wrapper.update();
                expect(wrapper.prop('griddingLoading')).toBeFalsy();
            });

            it('sholud pass the grid to the component', () => {
                const wrapper = shallow(<PlateEditorContainer {...props} />);
                expect(wrapper.prop('grid')).toEqual(grid);

            });

            it('should switch to the colony-detection step when onClickNext is called', () => {
                const wrapper = shallow(<PlateEditorContainer {...props} />);
                wrapper.prop('onClickNext')();
                wrapper.update();
                expect(wrapper.prop('step')).toEqual('colony-detection');
            });
        });
    });

    describe('colony detection', () => {
        it('should start the colony step with colony at position 0 0', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.setState({ step: 'colony-detection' });
            wrapper.update();
            expect(wrapper.prop('selectedColony')).toEqual({ row: 0, col: 0 });
        });

        it('should move to the next colony when onColonyFinish is called', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.setState({ step: 'colony-detection' });
            wrapper.prop('onColonyFinish')();
            wrapper.update();
            expect(wrapper.prop('selectedColony')).toEqual({ row: 0, col: 1 });
        });

        it('should move to the next row when necessary', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.setState({ step: 'colony-detection' });
            wrapper.prop('onColonyFinish')();
            wrapper.prop('onColonyFinish')();
            wrapper.update();
            expect(wrapper.prop('selectedColony')).toEqual({ row: 1, col: 0 });
        });

        it('should call the onFinish callback when onClickNext is called', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.setState({ step: 'colony-detection' });
            wrapper.prop('onClickNext')();
            expect(props.onFinish).toHaveBeenCalled();
        });

        it('should call the onFinish callback after the last colony', () => {
            const wrapper = shallow(<PlateEditorContainer {...props} />);
            wrapper.setState({ step: 'colony-detection' });
            wrapper.setState({ selectedColony: { row: 2, col: 1 } });
            wrapper.prop('onColonyFinish')();
            wrapper.update();
            expect(props.onFinish).toHaveBeenCalled();
        });
    });

});
