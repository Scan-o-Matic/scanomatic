import { shallow } from 'enzyme';
import React from 'react';

import './enzyme-setup';
import Gridding from '../../src/components/Gridding';

describe('<Gridding />', () => {
    const props = {
        loading: false,
        onRegrid: jasmine.createSpy('onRegrid'),
        rowOffset: 1,
        colOffset: 2,
        onRowOffsetChange: jasmine.createSpy('onRowOffsetChange'),
        onColOffsetChange: jasmine.createSpy('onColOffsetChange'),
    };

    it('should render a title', () => {
        const wrapper = shallow(<Gridding {...props} />);
        expect(wrapper.find('h4').text()).toEqual('Gridding');
    });

    it('should render a Re-grid <Button />', () => {
        const wrapper = shallow(<Gridding {...props} />);
        expect(wrapper.find('form .btn-regrid').exists()).toBe(true);
        expect(wrapper.find('form .btn-regrid').text()).toEqual('Re-grid');
    });

    it('should call onRegrid when Re-grid button is clicked', () => {
        const wrapper = shallow(<Gridding {...props} />);
        wrapper.find('form .btn-regrid').simulate('click');
        expect(props.onRegrid).toHaveBeenCalled();
    });

    it('should render a number input for the row offset', () => {
        const wrapper = shallow(<Gridding {...props} />);
        const input = wrapper.find('input.row-offset');
        expect(input.exists()).toBe(true);
        expect(input.prop('type')).toEqual('number');
        expect(input.prop('value')).toEqual(props.rowOffset);

    });

    it('should call onRowOffsetChange when the row offset is changed', () => {
        const wrapper = shallow(<Gridding {...props} />);
        wrapper.find('input.row-offset')
            .simulate('change', { target: { value: '42' } });
        expect(props.onRowOffsetChange).toHaveBeenCalledWith(42);
    });

    it('should render a number input for the col offset', () => {
        const wrapper = shallow(<Gridding {...props} />);
        const input = wrapper.find('input.col-offset');
        expect(input.exists()).toBe(true);
        expect(input.prop('type')).toEqual('number');
        expect(input.prop('value')).toEqual(props.colOffset);

    });

    it('should call onColOffsetChange when the col offset is changed', () => {
        const wrapper = shallow(<Gridding {...props} />);
        wrapper.find('input.col-offset')
            .simulate('change', { target: { value: '42' } });
        expect(props.onColOffsetChange).toHaveBeenCalledWith(42);
    });

    it('should render the error as alert-danger', () => {
        const wrapper = shallow(<Gridding {...props} error="XxX" />);
        expect(wrapper.find('form .alert-danger').text()).toContain('XxX');
    });

    it('should render an alert-success if no error', () => {
        const wrapper = shallow(<Gridding {...props} />);
        expect(wrapper.find('form .alert-success').text())
            .toContain('Gridding was succesful!');
    });

    describe('loading state', () => {
        it('should render a progress bar', () => {
            const wrapper = shallow(<Gridding {...props} loading />);
            expect(wrapper.find('div.progress').exists()).toBeTruthy();
        });

        it('should hide the form', () => {
            const wrapper = shallow(<Gridding {...props} loading />);
            expect(wrapper.find('form').exists()).toBeFalsy();
        });
    });
});
