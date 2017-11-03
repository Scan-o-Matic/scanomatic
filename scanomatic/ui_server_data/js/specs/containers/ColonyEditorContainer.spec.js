import { mount, shallow } from 'enzyme';
import React from 'react';

import '../components/enzyme-setup';
import ColonyEditorContainer from '../../ccc/containers/ColonyEditorContainer';
import colonyData from '../fixtures/colonyData.json';
import * as API from '../../ccc/api';

describe('</ColonyEditorContainer />', () => {
    const props = {
        accessToken: 'T0P53CR3T',
        ccc: 'CCC42',
        col: 1,
        image: '1M4G3',
        plate: 'PL4T3',
        row: 4,
        onFinish: jasmine.createSpy('onFinish'),
    };

    beforeEach(() => {
        props.onFinish.calls.reset();
        spyOn(API, 'SetColonyDetection').and.callFake(
            (ccc, image, plate, accessToken, row, col, onSuccess) => {
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
            props.ccc, props.image, props.plate,
            props.accessToken, props.row, props.col, jasmine.any(Function),
            jasmine.any(Function));
    });

    it('should reload data from the API when props are updated', () => {
        const wrapper = mount(<ColonyEditorContainer {...props}/>);
        wrapper.setProps({ col: 2 });
        expect(API.SetColonyDetection).toHaveBeenCalledWith(
            props.ccc, props.image, props.plate,
            props.accessToken, props.row, 2, jasmine.any(Function),
            jasmine.any(Function));
    });

    it('should pass the loaded data to the <ColonyEditor />', () => {
        const wrapper = mount(<ColonyEditorContainer {...props}/>);
        expect(wrapper.find('ColonyEditor').prop('data'))
            .toEqual(colonyData);
    });

    describe('#handleSet', () => {
        it('should send the data to the server', () => {
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(API.SetColonyCompression).toHaveBeenCalledWith(
                props.ccc, props.image, props.plate,
                props.accessToken, colonyData, props.row, props.col,
                jasmine.any(Function), jasmine.any(Function),
            );
        });

        it('should call the onFinish callback on success', () => {
            API.SetColonyCompression.and.callFake(
                (ccc, image, plate, accessToken, row, col, data, onSuccess) => {
                    onSuccess();
                }
            );
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(props.onFinish).toHaveBeenCalled();
        });

        it('should show an alert on error', () => {
            API.SetColonyCompression.and.callFake(
                (ccc, image, plate, accessToken, row, col, data, onSuccess, onError) => {
                    onError({ reason: 'Whoops' });
                }
            );
            spyOn(window, 'alert');
            const wrapper = mount(<ColonyEditorContainer {...props}/>);
            wrapper.find('ColonyEditor').prop('onSet')();
            expect(window.alert)
                .toHaveBeenCalledWith("Set Colony compression Error: Whoops");

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
