import React from 'react';
import { shallow } from 'enzyme';

import './enzyme-setup';
import Gridding from '../../ccc/components/Gridding';

describe('<Gridding />', () => {
    let props;

    beforeEach(() => {
        props = {
            status: 'error',
            alert: 'The foobar is broken',
            image: new Image,
            colOffset: 1,
            rowOffset: 2,
            onRegrid: jasmine.createSpy('onRegrid'),
            onNext: jasmine.createSpy('onNext'),
            onRowOffsetChange: jasmine.createSpy('onRowOffsetChange'),
            onColOffsetChange: jasmine.createSpy('onColOffsetChange'),
        };
    });

    it('should render a <PlateContainer />', () => {
        const wrapper = shallow(<Gridding {...props} />);
        expect(wrapper.find('PlateContainer').exists()).toBe(true);
    });

    it('should render a Re-grid <Button />', () => {
        const wrapper = shallow(<Gridding {...props} />);
        expect(wrapper.find('.btn-regrid').exists()).toBe(true);
        expect(wrapper.find('.btn-regrid').text()).toEqual('Re-grid');
    });

    it('should still render a Re-grid <Button /> if success', () => {
        const wrapper = shallow(<Gridding {...props} status="success" />);
        expect(wrapper.find('.btn-regrid').exists()).toBe(true);
        expect(wrapper.find('.btn-regrid').text()).toEqual('Re-grid');
    });

    it('should call onRegrid when Re-grid button is clicked', () => {
        const wrapper = shallow(<Gridding {...props} />);
        wrapper.find('.btn-regrid').simulate('click');
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
        wrapper.find('input.row-offset').simulate('change', {target: {value: 42}});
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
        wrapper.find('input.col-offset').simulate('change', { target: { value: 42 } });
        expect(props.onColOffsetChange).toHaveBeenCalledWith(42);
    });

    it('should not render a "Next Step" <Button /> if no success', () => {
        const wrapper = shallow(<Gridding {...props} status="error" />);
        expect(wrapper.find('.btn-next').exists()).toBe(false);
    });

    it('should render a "Next Step" <Button /> if success', () => {
        const wrapper = shallow(<Gridding {...props} status="success" />);
        expect(wrapper.find('.btn-next').exists()).toBe(true);
        expect(wrapper.find('.btn-next').text()).toEqual('Next Step');
    });

    it('should call onNext when the "Next Step" button is clicked', () => {
        const wrapper = shallow(<Gridding {...props} status="success" />);
        wrapper.find('.btn-next').simulate('click');
        expect(props.onNext).toHaveBeenCalled();
    });

    it('should render the passed alert', () => {
        const wrapper = shallow(<Gridding {...props} />);
        expect(wrapper.find('.alert').exists()).toBe(true);
        expect(wrapper.find('.alert').text()).toEqual(props.alert);
    });

    it('should disable the regrid button when loading', () => {
        const wrapper = shallow(<Gridding {...props} status="loading" />);
        expect(wrapper.find('.btn-regrid').prop('disabled')).toBeTruthy();

    });
});
