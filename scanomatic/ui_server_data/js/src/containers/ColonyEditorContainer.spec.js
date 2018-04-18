import { mount, shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ColonyEditorContainer from '../../src/containers/ColonyEditorContainer';
import colonyData from '../fixtures/colonyData.json';
import * as API from '../../src/api';

describe('</ColonyEditorContainer />', () => {
    const props = {
        accessToken: 'T0P53CR3T',
        ccc: 'CCC42',
        col: 1,
        image: '1M4G3',
        plateId: 1,
        row: 4,
        onFinish: jasmine.createSpy('onFinish'),
    };

    beforeEach(() => {
        props.onFinish.calls.reset();
        spyOn(API, 'SetColonyDetection').and.callFake(
            (ccc, image, plateId, accessToken, row, col, onSuccess) => {
                onSuccess({
                    image: colonyData.image,
                    image_min: colonyData.imageMin,
                    image_max: colonyData.imageMax,
                    blob: colonyData.blob,
                    background: colonyData.background,
                });
            }
        );
        spyOn(API, 'SetColonyCompression')    });

    it('should render a <ColonyEditor />', () => {
        const wrapper = shallow(<ColonyEditorContainer {...props} />);
        expect(wrapper.find('ColonyEditor').exists()).toBe(true);
    });

    it('should load the colony data from the api', () => {
        mount(<ColonyEditorContainer {...props}/>);
        expect(API.SetColonyDetection).toHaveBeenCalledWith(
            props.ccc, props.image, props.plateId,
            props.accessToken, props.row, props.col, jasmine.any(Function),
            jasmine.any(Function));
    });

    it('should reload data from the API when props are updated', () => {
        const wrapper = mount(<ColonyEditorContainer {...props}/>);
        wrapper.setProps({ col: 2 });
        expect(API.SetColonyDetection).toHaveBeenCalledWith(
            props.ccc, props.image, props.plateId,
            props.accessToken, props.row, 2, jasmine.any(Function),
            jasmine.any(Function));
    });

    it('should reset the cellCount and cellCountError state when props are updated', () => {
        const wrapper = mount(<ColonyEditorContainer {...props}/>);
        wrapper.setState({ cellCount: 33, cellCountError: true });
        wrapper.setProps({ col: 2 });
        expect(wrapper.state('cellCount')).toBe(null);
        expect(wrapper.state('cellCountError')).toBeFalsy();
    });

    it('should pass the loaded data to the <ColonyEditor />', () => {
        const wrapper = mount(<ColonyEditorContainer {...props}/>);
        expect(wrapper.find('ColonyEditor').prop('data'))
            .toEqual(colonyData);
    });

    describe('#handleCellCountChange', () => {
        it('should set cell count', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onCellCountChange')(666);
            wrapper.update();
            expect(wrapper.find('ColonyEditor').prop('cellCount')).toEqual(666);
        });

        it('should set cellCountError to true if cell count is < 0', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onCellCountChange')(-666);
            wrapper.update();
            expect(wrapper.find('ColonyEditor').prop('cellCountError')).toBeTruthy();
        });

        it('should set cellCountError to false if cell count is > 0', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onCellCountChange')(666);
            expect(wrapper.find('ColonyEditor').prop('cellCountError')).toBeFalsy();
        });

        it('should set cellCountError to false if cell count is = 0', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onCellCountChange')(0);
            expect(wrapper.find('ColonyEditor').prop('cellCountError')).toBeFalsy();
        });
    });

    describe('#handleSet', () => {
        it('should not call the API if cell count not set', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(API.SetColonyCompression).not.toHaveBeenCalled()
        });

        it('should set cellCountError to true if cell count is not set', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onSet')();
            wrapper.update();
            expect(wrapper.find('ColonyEditor').prop('cellCountError')).toBeTruthy();
        });

        it('should not call the API if cell count error is true', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.setState({ cellCount: -666 });
            wrapper.setState({ cellCountError: true });
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(API.SetColonyCompression).not.toHaveBeenCalled()
        });

        it('should send the data to the server', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.setState({ cellCount: 666 });
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(API.SetColonyCompression).toHaveBeenCalledWith(
                props.ccc, props.image, props.plateId, props.accessToken,
                colonyData, 666, props.row, props.col,
                jasmine.any(Function), jasmine.any(Function),
            );
        });

        it('should call the onFinish callback on success', () => {
            API.SetColonyCompression.and.callFake(
                (ccc, image, plateId, accessToken, row, col, data, cellCount, onSuccess) => {
                    onSuccess();
                }
            );
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.setState({ cellCount: 666 });
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(props.onFinish).toHaveBeenCalled();
        });

        it('should show an alert on error', () => {
            API.SetColonyCompression.and.callFake(
                (ccc, image, plateId, accessToken, row, col, data, cellCount, onSuccess, onError) => {
                    onError({ reason: 'Whoops' });
                }
            );
            spyOn(window, 'alert');
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.setState({ cellCount: 666 });
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(window.alert)
                .toHaveBeenCalledWith("Set Colony compression Error: Whoops");
        });
    });

    describe('#handleSkip', () => {
        it('should call the onFinish callback', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onSkip')();
            expect(props.onFinish).toHaveBeenCalled();
        });
    });

    describe('#handleUpdate', () => {
        const newData = {
            blob: [[true, true], [false, false]],
            background: [[false, false], [true, true]],
        };

        it('should update the state', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.children().prop('onUpdate')(newData);
            wrapper.update();
            expect(wrapper.state('colonyData').blob).toEqual(newData.blob);
            expect(wrapper.state('colonyData').background).toEqual(newData.background);
        });
    });
});
