import React from 'react';
import { mount } from 'enzyme';

import '../components/enzyme-setup';
import GriddingContainer from '../../ccc/containers/GriddingContainer.js';
import * as API from '../../ccc/api';

describe('<GriddingContainer />', () => {
    const props = {
        accessToken: 'T0P53CR3T',
        cccId: 'CCC42',
        imageId: '1M4G3',
        plateId: 'PL4T3',
        pinFormat: [8, 12],
        onFinish: jasmine.createSpy('onFinish'),
    };

    beforeEach(() => {
        props.onFinish.calls.reset();
        spyOn(API, 'SetGridding');
    });

    it('should render a <Gridding />', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        expect(wrapper.find('Gridding').exists()).toBe(true);
    });

    it('should set the row/col offsets to 4/0', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        expect(wrapper.children().prop('rowOffset')).toEqual(4);
        expect(wrapper.children().prop('colOffset')).toEqual(0);
    });

    it('should update its state when row offset change', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        wrapper.children().prop('onRowOffsetChange')(42)
        expect(wrapper.state('rowOffset')).toEqual(42);
    });

    it('should update its state when col offset change', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        wrapper.children().prop('onColOffsetChange')(2)
        expect(wrapper.state('colOffset')).toEqual(2);
    });

    it('should get the grid from the API', () => {
        mount(<GriddingContainer {...props} />);
        expect(API.SetGridding).toHaveBeenCalledWith(
            props.cccId, props.imageId, props.plateId, props.pinFormat,
            [0, 0], props.accessToken, jasmine.any(Function), jasmine.any(Function)
        );
    });

    it('should set the alert to "Calculating Gridding ... please wait ...!"', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        expect(wrapper.children().prop('alert'))
            .toEqual('Calculating Gridding ... please wait ...!');
    });

    it('should set status to "loading"', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        expect(wrapper.children().prop('status')).toEqual('loading');
    });

    describe('when gridding succeeds', () => {
        const data = { grid: [[[0]], [[0]]], };

        beforeEach(() => {
            API.SetGridding.and.callFake(
                (cccId, imageId, plateId, pinFormat, offsets, accessToken, onSuccess) => {
                    onSuccess(data);
                }
            );
        });

        it('should pass the received grid to its child on success', () => {
            const wrapper = mount(<GriddingContainer {...props} />);
            expect(wrapper.children().prop('grid')).toEqual(data.grid);
        });

        it('should set the alert to "Gridding was sucessful!"', () => {
            const wrapper = mount(<GriddingContainer {...props} />);
            expect(wrapper.children().prop('alert'))
                .toEqual('Gridding was succesful!');
        });

        it('should set status to "success"', () => {
            const wrapper = mount(<GriddingContainer {...props} />);
            expect(wrapper.children().prop('status')).toEqual('success');
        });
    });

    describe('when gridding fails', () => {
        const data = {
            grid: [[[0]], [[0]]],
            reason: "Stuff's broken",
        };

        beforeEach(() => {
            API.SetGridding.and.callFake(
                (cccId, imageId, plateId, pinFormat, offsets, accessToken, onSuccess, onError) => {
                    onError(data);
                }
            );
        });

        it('should pass the received grid to its child on success', () => {
            const wrapper = mount(<GriddingContainer {...props} />);
            expect(wrapper.children().prop('grid')).toEqual(data.grid);
        });

        it('should set the alert to "Gridding was unsuccesful. Reason: ..."', () => {
            const wrapper = mount(<GriddingContainer {...props} />);
            expect(wrapper.children().prop('alert'))
                .toEqual("Gridding was unsuccesful. Reason: 'Stuff's broken'. Please enter Offset and retry!");
        });

        it('should set status to "error"', () => {
            const wrapper = mount(<GriddingContainer {...props} />);
            expect(wrapper.children().prop('status')).toEqual('error');
        });
    });

    it('should get a new grid using the offsets when onRegrid is called', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        wrapper.setState( { rowOffset: 2, colOffset: 3 });
        wrapper.children().prop('onRegrid')();
        expect(API.SetGridding).toHaveBeenCalledWith(
            props.cccId, props.imageId, props.plateId,
            props.pinFormat, [2, 3], props.accessToken,
            jasmine.any(Function), jasmine.any(Function)
        );
    });

    it('should call onFinish when children calls onNext', () => {
        const wrapper = mount(<GriddingContainer {...props} />);
        console.log(wrapper.children().props());
        wrapper.children().prop('onNext')();
        expect(props.onFinish).toHaveBeenCalled();
    });
});
